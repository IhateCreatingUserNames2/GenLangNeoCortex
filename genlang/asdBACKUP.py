import os
import numpy as np
import json
import logging
from typing import List, Tuple, Dict, Optional, TypedDict
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
import random

# LangChain / LangGraph Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field as PydanticField
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END

# Carregar variáveis de ambiente (para a API key)
from dotenv import load_dotenv

load_dotenv()

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==========================
# ESTRUTURAS DE DADOS BASE
# ==========================

@dataclass
class GenlangVector:
    vector: np.ndarray
    source_text: str
    source_agent: str

    def similarity(self, other: 'GenlangVector') -> float:
        cos_sim = np.dot(self.vector, other.vector) / (np.linalg.norm(self.vector) * np.linalg.norm(other.vector))
        return float(cos_sim)


@dataclass
class MathTask:
    task_id: int
    task_type: str
    problem_statement: str
    expected_output: float
    complexity: int = 1


class CognitiveState(Enum):
    RIGID = "rigid"
    OPTIMAL = "optimal"
    CHAOTIC = "chaotic"


# ==========================
# CAMADA 1: AGENTES COMUNICADORES (COM PENSAMENTO VETORIAL)
# ==========================

class LLMGenlangAgent:
    """Agente LLM que se comunica gerando texto e convertendo para vetores Genlang,
    com capacidade de modificar vetores de pensamento existentes."""

    def __init__(self, agent_id: str, specialization: str, embedding_model: OpenAIEmbeddings,
                 vector_modification_ratio: float = 0.5):
        self.agent_id = agent_id
        self.specialization = specialization
        self.embedding_model = embedding_model
        self.vector_modification_ratio = vector_modification_ratio

        # O LLM é passado dinamicamente para permitir a mudança de temperatura
        self.llm: Optional[ChatOpenAI] = None

        # Prompt para gerar um novo pensamento do zero
        self.creation_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Você é o Agente {agent_id}, especialista em {specialization}. Gere um pensamento curto e conciso para o próximo passo na solução do problema matemático, baseado no histórico."),
            ("human", "Problema: '{problem_statement}'\nHistórico:\n{conversation_history}\n\nSeu pensamento:")
        ])

        # Prompt para modificar um pensamento existente (alinhado ao framework)
        self.modification_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Você é um controlador de pensamento vetorial. Descreva verbalmente a *modificação* que o vetor de pensamento atual precisa para se aproximar da solução. Use termos como 'adicionar conceito X', 'aumentar importância Y', 'correlacionar com operação Z'."),
            ("human",
             "Problema: '{problem_statement}'\nContexto Atual: '{context_summary}'\n\nSua descrição da modificação:")
        ])

    def set_llm(self, llm: ChatOpenAI):
        """Define a instância do LLM a ser usada, permitindo temperatura dinâmica."""
        self.llm = llm

    def generate_concept(self, task: MathTask, context: List[GenlangVector]) -> GenlangVector:
        if self.llm is None:
            raise ValueError("LLM não foi definido para o agente. Chame set_llm() primeiro.")

        # Decide se vai criar um novo conceito ou modificar o existente
        should_modify = context and random.random() < self.vector_modification_ratio

        if should_modify:
            # --- MODO DE MODIFICAÇÃO VETORIAL ---
            context_summary = " -> ".join([c.source_text for c in context[-3:]])
            context_vector = np.mean([c.vector for c in context], axis=0)

            chain = self.modification_prompt | self.llm
            response = chain.invoke({"problem_statement": task.problem_statement, "context_summary": context_summary})
            modification_text = response.content

            delta_vector = np.array(self.embedding_model.embed_query(modification_text))
            new_vector = context_vector + delta_vector
            new_vector /= np.linalg.norm(new_vector)

            source_text = f"[MOD] {modification_text}"
            logger.debug(f"Agente {self.agent_id} modificou o pensamento com: '{modification_text}'")

            return GenlangVector(vector=new_vector, source_text=source_text, source_agent=self.agent_id)
        else:
            # --- MODO DE CRIAÇÃO ---
            history_str = "\n".join(
                [f"- {c.source_agent}: '{c.source_text}'" for c in context]) or "(Nenhuma conversa ainda)"

            chain = self.creation_prompt | self.llm
            response = chain.invoke({
                "agent_id": self.agent_id,
                "specialization": self.specialization,
                "problem_statement": task.problem_statement,
                "conversation_history": history_str
            })
            thought_text = response.content

            vector = np.array(self.embedding_model.embed_query(thought_text))
            logger.debug(f"Agente {self.agent_id} criou o pensamento: '{thought_text}'")

            return GenlangVector(vector=vector, source_text=thought_text, source_agent=self.agent_id)


# ==========================
# CAMADA 2: GOVERNADOR DE CRITICALIDADE
# ==========================

class CriticalityGovernor:
    """Monitora o Edge of Coherence e fornece recomendações para governança."""

    def __init__(self, history_size: int = 100):
        self.communication_history = deque(maxlen=history_size)
        self.task_success_history = deque(maxlen=history_size)

    def record_cycle(self, exchanges: List[GenlangVector], success: bool):
        self.communication_history.extend(exchanges)
        self.task_success_history.append(float(success))

    def assess_and_govern(self, current_temp: float) -> Tuple[float, CognitiveState, Dict[str, float]]:
        """Avalia o estado e recomenda uma nova temperatura."""
        if len(self.communication_history) < 20:
            return current_temp, CognitiveState.OPTIMAL, {"novelty": 0.5, "coherence": 0.5, "grounding": 0.5}

        metrics = {
            "novelty": self._calculate_novelty(),
            "coherence": self._calculate_coherence(),
            "grounding": self._calculate_grounding()
        }

        # Lógica de Governança
        new_temp = current_temp
        if metrics["novelty"] < 0.3 and metrics["coherence"] > 0.6:
            state = CognitiveState.RIGID
            new_temp = min(current_temp * 1.2, 1.3)
            logger.warning(f"Estado RÍGIDO detectado. Aumentando temperatura para {new_temp:.2f}")
        elif metrics["novelty"] > 0.7 and metrics["coherence"] < 0.3:
            state = CognitiveState.CHAOTIC
            new_temp = max(current_temp * 0.8, 0.1)
            logger.warning(f"Estado CAÓTICO detectado. Reduzindo temperatura para {new_temp:.2f}")
        else:
            state = CognitiveState.OPTIMAL

        return new_temp, state, metrics

    # Funções de cálculo de métrica permanecem as mesmas...
    def _calculate_novelty(self) -> float:
        if len(self.communication_history) < 20: return 0.5
        recent, older = list(self.communication_history)[-10:], list(self.communication_history)[:-10]
        if not older: return 0.8
        distances = [1 - r.similarity(o) for r in recent for o in older]
        return np.mean(distances) if distances else 0.5

    def _calculate_coherence(self) -> float:
        if len(self.communication_history) < 2: return 0.5
        recent = list(self.communication_history)[-20:]
        similarities = [recent[i].similarity(recent[i + 1]) for i in range(len(recent) - 1)]
        return np.mean(similarities) if similarities else 0.5

    def _calculate_grounding(self) -> float:
        if not self.task_success_history: return 0.5
        return np.mean(list(self.task_success_history)[-50:])


# ==========================
# CAMADA 3: MEMÓRIA E META-APRENDIZADO
# ==========================

class MemorySystem:
    """Sistema de memória que agora pode rastrear a evolução dos conceitos."""

    def __init__(self, cluster_threshold: float = 0.8):
        self.concept_clusters: Dict[str, List[GenlangVector]] = {}
        self.cluster_threshold = cluster_threshold

    def store_communication(self, exchange: List[GenlangVector]):
        for vector in exchange:
            self._update_concept_clusters(vector)

    def _update_concept_clusters(self, vector: GenlangVector):
        if not self.concept_clusters:
            self.concept_clusters["concept_0"] = [vector]
            return

        similarities = {cid: np.mean([vector.similarity(v) for v in c_vectors]) for cid, c_vectors in
                        self.concept_clusters.items()}
        best_cluster, best_sim = max(similarities.items(), key=lambda item: item[1])

        if best_sim > self.cluster_threshold:
            self.concept_clusters[best_cluster].append(vector)
        else:
            new_id = f"concept_{len(self.concept_clusters)}"
            self.concept_clusters[new_id] = [vector]

    def get_vocabulary_size(self) -> int:
        return len(self.concept_clusters)

    def get_cluster_centroids(self) -> Dict[str, np.ndarray]:
        """Calcula o vetor médio (centroide) para cada cluster."""
        return {cid: np.mean([v.vector for v in vectors], axis=0) for cid, vectors in self.concept_clusters.items() if
                vectors}


# ==========================
# SISTEMA PRINCIPAL COM LANGGRAPH E GOVERNANÇA ATIVA
# ==========================

# Estado do grafo, agora com temperatura
class GraphState(TypedDict):
    task: MathTask
    exchanges: List[GenlangVector]
    current_agent_idx: int
    max_exchanges: int
    llm_temperature: float
    solution: Optional[float]
    success: bool
    error: Optional[float]


class EmergentGenlangSystem:
    def __init__(self, vector_dim: int = 256, max_exchanges: int = 6, initial_temp: float = 0.5):
        self.max_exchanges = max_exchanges
        self.initial_temp = initial_temp

        # Modelos
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=vector_dim)

        # Componentes do Sistema
        self.agents = [
            LLMGenlangAgent("Agente_A", "Analista de Problemas", self.embedding_model),
            LLMGenlangAgent("Agente_B", "Calculador", self.embedding_model),
            LLMGenlangAgent("Agente_C", "Verificador", self.embedding_model)
        ]
        self.governor = CriticalityGovernor()
        self.memory = MemorySystem()

        # Histórico para Análise
        self.task_history: List[Dict] = []
        self.metrics_history: List[Dict] = []
        self.centroid_history: List[Dict] = []

        # Grafo
        self.graph = self._build_graph()

    def _get_llm_instance(self, temperature: float) -> ChatOpenAI:
        """Cria uma instância do LLM com a temperatura desejada."""
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        def agent_node(state: GraphState) -> GraphState:
            agent = self.agents[state["current_agent_idx"]]
            llm_instance = self._get_llm_instance(state["llm_temperature"])
            agent.set_llm(llm_instance)

            concept = agent.generate_concept(state["task"], state["exchanges"])
            state["exchanges"].append(concept)
            state["current_agent_idx"] = (state["current_agent_idx"] + 1) % len(self.agents)
            return state

        def interpreter_node(state: GraphState) -> GraphState:
            conversation_str = "\n".join([f"- {c.source_agent}: '{c.source_text}'" for c in state["exchanges"]])

            class NumericSolution(BaseModel):
                answer: Optional[float] = PydanticField(
                    description="A resposta numérica final. Use null se não houver.")

            parser = PydanticOutputParser(pydantic_object=NumericSolution)
            prompt = ChatPromptTemplate.from_template(
                "Analise a conversa para resolver o problema. Extraia a resposta numérica final em JSON.\n"
                "Problema: {problem}\nConversa:\n{conversation}\n{format_instructions}"
            )
            chain = prompt | self._get_llm_instance(0.1) | parser  # Usar baixa temp para extração

            try:
                result = chain.invoke({"problem": state["task"].problem_statement, "conversation": conversation_str,
                                       "format_instructions": parser.get_format_instructions()})
                solution = result.answer
            except Exception:
                solution = None

            state["success"] = solution is not None and abs(solution - state["task"].expected_output) < 0.1
            state["solution"] = solution
            state["error"] = abs(solution - state["task"].expected_output) if solution is not None else None
            return state

        def should_continue(state: GraphState) -> str:
            return "end" if state["success"] or len(state["exchanges"]) >= state["max_exchanges"] else "continue"

        workflow.add_node("agent", agent_node)
        workflow.add_node("interpreter", interpreter_node)
        workflow.set_entry_point("agent")
        workflow.add_edge("agent", "interpreter")
        workflow.add_conditional_edges("interpreter", should_continue, {"continue": "agent", "end": END})
        return workflow.compile()

    def train(self, num_epochs: int = 20, tasks_per_epoch: int = 10):
        epoch_state = {"llm_temperature": self.initial_temp}

        for epoch in range(num_epochs):
            for i in range(tasks_per_epoch):
                task_id = epoch * tasks_per_epoch + i
                complexity = min(epoch // 4 + 1, 5)
                task = self._generate_task(task_id, complexity)
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Task {i + 1}/{tasks_per_epoch}: {task.problem_statement}")

                initial_state = GraphState(
                    task=task, exchanges=[], current_agent_idx=0, max_exchanges=self.max_exchanges,
                    llm_temperature=epoch_state["llm_temperature"],
                    solution=None, success=False, error=None
                )
                final_state = self.graph.invoke(initial_state)

                self.task_history.append(
                    {'epoch': epoch, 'success': final_state['success'], 'exchanges': len(final_state['exchanges'])})
                self.governor.record_cycle(final_state['exchanges'], final_state['success'])
                self.memory.store_communication(final_state['exchanges'])

            # Governança e registro no final da época
            new_temp, cog_state, metrics = self.governor.assess_and_govern(epoch_state["llm_temperature"])
            epoch_state["llm_temperature"] = new_temp

            self.metrics_history.append({**metrics, 'epoch': epoch, 'temperature': new_temp})
            self.centroid_history.append({'epoch': epoch, 'centroids': self.memory.get_cluster_centroids()})

            vocab_size = self.memory.get_vocabulary_size()
            success_rate = np.mean([h['success'] for h in self.task_history if h['epoch'] == epoch])
            logger.info(
                f"Epoch {epoch + 1} Resumo: Sucesso: {success_rate:.2f}, Vocabulário: {vocab_size}, Estado: {cog_state.value}, Temp: {new_temp:.2f}")

    def _generate_arithmetic_task(self, complexity: int) -> Tuple[str, float]:
        """Gera uma tarefa de aritmética básica."""
        a = np.random.randint(1, 10 * complexity)
        b = np.random.randint(1, 10 * complexity)
        op = random.choice(['+', '-', '*', '/'])
        if op == '/' and b == 0: b = 1

        problem = f"{a} {op} {b}"
        result = eval(problem)
        problem_statement = f"Calcule: {problem}"
        return problem_statement, float(result)

    def _generate_algebra_task(self, complexity: int) -> Tuple[str, float]:
        """Gera uma tarefa de álgebra linear simples (ax + b = c)."""
        a = np.random.randint(1, 5 * complexity)
        x = np.random.randint(-10, 10)
        b = np.random.randint(-20, 20)
        c = a * x + b
        problem_statement = f"Resolva para x: {a}x + {b} = {c}"
        return problem_statement, float(x)

    def _generate_noisy_task(self, complexity: int) -> Tuple[str, float]:
        """Gera uma tarefa com informações irrelevantes."""
        # A tarefa principal é uma simples operação aritmética
        a = np.random.randint(1, 10 * complexity)
        b = np.random.randint(1, 10 * complexity)
        result = a + b  # Vamos manter a operação simples (soma) para isolar o desafio do ruído

        # Adicionar informações de ruído
        num_wagons = random.randint(5, 20)
        temperature = random.randint(0, 30)
        color = random.choice(["azul", "vermelho", "verde"])

        problem_statement = (
            f"Um trem com {num_wagons} vagões da cor {color} "
            f"transporta {a} passageiros. Na próxima estação, mais {b} passageiros embarcam. "
            f"A temperatura é de {temperature}°C. Quantos passageiros estão no trem agora?"
        )
        return problem_statement, float(result)

    def _generate_multistep_logic_task(self, complexity: int) -> Tuple[str, float]:
        """Gera uma tarefa de lógica que requer múltiplos passos."""
        # Exemplo: A = B + C; D = A * E
        b = random.randint(1, 5 * complexity)
        c = random.randint(1, 5 * complexity)
        e = random.randint(2, 4)

        a = b + c
        d = a * e

        problem_statement = (
            f"O valor inicial de A é a soma de {b} e {c}. "
            f"O resultado final é o valor de A multiplicado por {e}. "
            f"Qual é o resultado final?"
        )
        return problem_statement, float(d)

    def _generate_task(self, task_id: int, complexity: int) -> MathTask:
        """Gera uma tarefa matemática com complexidade e tipo crescentes."""

        # Mapeamento de complexidade para tipos de tarefas disponíveis
        # Nível 1-2: Apenas Aritmética e Álgebra
        # Nível 3: Adiciona problemas com ruído
        # Nível 4+: Adiciona problemas de múltiplos passos

        available_tasks = [('arithmetic', self._generate_arithmetic_task)]
        if complexity >= 1:
            available_tasks.append(('algebra', self._generate_algebra_task))
        if complexity >= 3:
            available_tasks.append(('noisy_logic', self._generate_noisy_task))
        if complexity >= 4:
            available_tasks.append(('multistep_logic', self._generate_multistep_logic_task))

        task_type_str, task_func = random.choice(available_tasks)

        problem_statement, expected_output = task_func(complexity)

        return MathTask(
            task_id=task_id,
            task_type=task_type_str,
            problem_statement=problem_statement,
            expected_output=expected_output,
            complexity=complexity
        )

    def plot_results(self):
        if not self.task_history:
            logger.warning("Nenhum histórico para plotar.");
            return

        df_tasks = pd.DataFrame(self.task_history)
        df_metrics = pd.DataFrame(self.metrics_history)

        fig, axes = plt.subplots(4, 1, figsize=(14, 22), sharex=True)
        fig.suptitle("Análise da Evolução do Sistema Genlang", fontsize=16)

        # 1. Taxa de Sucesso
        df_epoch_summary = df_tasks.groupby('epoch')['success'].mean().reset_index()
        axes[0].plot(df_epoch_summary['epoch'], df_epoch_summary['success'], 'o-', label='Taxa de Sucesso', alpha=0.7)
        axes[0].plot(df_epoch_summary['epoch'], df_epoch_summary['success'].rolling(window=5, min_periods=1).mean(),
                     'r-', label='Média Móvel')
        axes[0].set_ylabel("Taxa de Sucesso");
        axes[0].set_title("Desempenho da Tarefa");
        axes[0].grid(True);
        axes[0].legend()

        # 2. Vocabulário e Eficiência
        ax2_twin = axes[1].twinx()

        # <<< LINHA CORRIGIDA AQUI >>>
        df_vocab = pd.Series({h['epoch']: len(h['centroids']) for h in self.centroid_history}).reset_index(
            name='vocab_size').rename(columns={'index': 'epoch'})

        df_efficiency = df_tasks.groupby('epoch')['exchanges'].mean().reset_index()
        axes[1].plot(df_vocab['epoch'], df_vocab['vocab_size'], 'g-s', label='Tamanho do Vocabulário')
        ax2_twin.plot(df_efficiency['epoch'], df_efficiency['exchanges'], 'm-p', label='Média de Trocas (Eficiência)')
        axes[1].set_ylabel("Nº de Conceitos", color='g');
        ax2_twin.set_ylabel("Nº de Trocas", color='m')
        axes[1].set_title("Vocabulário Emergente e Eficiência Comunicativa");
        axes[1].grid(True)

        # Ajuste para a legenda combinada funcionar melhor
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

        # 3. Estado Cognitivo e Governança
        ax3_twin = axes[2].twinx()
        axes[2].plot(df_metrics['epoch'], df_metrics['novelty'], '-o', label='Novelty')
        axes[2].plot(df_metrics['epoch'], df_metrics['coherence'], '-s', label='Coherence')
        axes[2].plot(df_metrics['epoch'], df_metrics['grounding'], '-^', label='Grounding')
        ax3_twin.plot(df_metrics['epoch'], df_metrics['temperature'], ':r', label='LLM Temp')
        axes[2].set_ylabel("Valor da Métrica");
        ax3_twin.set_ylabel("Temperatura", color='r')
        axes[2].set_title("Evolução do Estado Cognitivo e Governança");
        axes[2].grid(True)

        lines1, labels1 = axes[2].get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

        # 4. Estabilidade Conceitual
        drifts = []
        epochs_for_drift = []
        if len(self.centroid_history) > 1:
            for i in range(1, len(self.centroid_history)):
                prev_c, curr_c = self.centroid_history[i - 1]['centroids'], self.centroid_history[i]['centroids']
                common_keys = set(prev_c.keys()) & set(curr_c.keys())
                epoch_drift = [np.linalg.norm(prev_c[k] - curr_c[k]) for k in common_keys]
                if epoch_drift:
                    drifts.append(np.mean(epoch_drift))
                    epochs_for_drift.append(self.centroid_history[i]['epoch'])

        if drifts:
            axes[3].plot(epochs_for_drift, drifts, '-x', color='purple', label='Instabilidade Conceitual')

        axes[3].set_xlabel("Época");
        axes[3].set_ylabel("Mudança Média do Centroide")
        axes[3].set_title("Estabilidade da Linguagem Emergente");
        axes[3].grid(True);
        axes[3].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()


# ==========================
# DEMONSTRAÇÃO
# ==========================
if __name__ == "__main__":
    system = EmergentGenlangSystem(vector_dim=256, max_exchanges=6, initial_temp=0.6)

    logger.info("Iniciando treinamento do Sistema Genlang Emergente com Governança Ativa...")
    logger.info("=" * 70)

    system.train(num_epochs=50, tasks_per_epoch=20)

    logger.info("\nTreinamento concluído. Gerando visualizações...")
    system.plot_results()