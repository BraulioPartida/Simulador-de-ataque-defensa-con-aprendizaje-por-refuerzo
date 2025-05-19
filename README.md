# Simulador-de-ataque-con-aprendizaje-por-refuerzo

**Descripción**

Este proyecto implementa un **simulador de ciberataques y contramedidas** basado en **Reinforcement Learning (RL)**. Modela una pequeña red de nodos con servicios y vulnerabilidades aleatorias, y permite entrenar agentes (DQN, PPO) que aprenden a atacar esa red para maximizar la exfiltración de datos y minimizar la detección.

**Características principales**

* Servicios y vulnerabilidades asignados aleatoriamente a cada nodo.
* Estado completo del entorno: nodos descubiertos, servicios detectados, vulnerabilidades conocidas, privilegios, recursos, nivel de alerta y datos exfiltrados.
* Acciones del agente: `scan`, `exploit`, `escalate`, `move`, `exfiltrate`, `cover_tracks`.
* Recompensas y penalizaciones diseñadas para guiar la política de ataque.
* Implementación de algoritmos DQN y PPO con Stable-Baselines3.
* Flujo completo de entrenamiento, evaluación y comparación de modelos.
* Visualización de métricas y resultados: recompensas, nivel de alerta, datos extraídos, nodos comprometidos, distribución de acciones.

---

## Instalación

1. Clona este repositorio:

   ```bash
   git clone https://github.com/tuUsuario/Simulador-RL-Cybersec.git
   cd Simulador-RL-Cybersec
   ```

2. Crea y activa un entorno virtual (opcional pero recomendado):

   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\\Scripts\\activate    # Windows
   ```

3. Instala dependencias:

   ```bash
   pip install -r requirements.txt
   ```

   > **requirements.txt** incluye:
   >
   > * gymnasium
   > * stable-baselines3
   > * numpy
   > * pandas
   > * matplotlib
   > * seaborn
   > * tqdm

---

## Estructura del proyecto

```text
├── env/                       # Definición de AttackerEnv
│   ├── __init__.py
│   └── attacker_env.py        # Lógica del simulador MDP
│
├── train_dqn.py               # Script de entrenamiento DQN
├── train_ppo.py               # Script de entrenamiento PPO
├── compare_models.py          # Evaluación y comparación de modelos
├── requirements.txt           # Dependencias Python
├── README.md                  # Documentación
└── evaluation_results/        # Resultados (logs, CSV, gráficos)
```

---

## Uso rápido

### Entrenar un modelo DQN

```bash
python train_dqn.py
```

### Entrenar un modelo PPO

```bash
python train_ppo.py
```

### Comparar agentes DQN vs PPO

```bash
python compare_models.py
```

Los scripts crean carpetas `models/` y `logs/`, guardan checkpoints, modelos finales y gráficos en `evaluation_results/`.

---

## Configuración de hiperparámetros

En los scripts de entrenamiento (`train_dqn.py`, `train_ppo.py`) encontrarás variables:

* **TRAIN\_TIMESTEPS**: número de timesteps de entrenamiento.
* **LEARNING\_RATE**, **BATCH\_SIZE**, **BUFFER\_SIZE** (DQN) o **N\_STEPS**, **N\_EPOCHS** (PPO).
* Parámetros de exploración (`exploration_fraction`, `ent_coef`, `clip_range`, etc.).
* Semilla (`SEED`) para reproducibilidad.

Puedes ajustar estos valores para experimentar con estrategias más agresivas o conservadoras.

---

## Resultados y visualizaciones

Después de entrenar y comparar, revisa la carpeta `evaluation_results/`:

* **comparison\_stats.json**: estadísticas agregadas de cada modelo.
* **all\_episodes\_data.csv**: datos crudos de cada episodio.
* **plots/**: gráficos comparativos:

  * `reward_comparison.png` (boxplot de recompensas)
  * `metrics_comparison.png` (barras de métricas clave)
  * `action_distribution.png` (frecuencia de acciones)
  * `nodes_progression.png` y `alert_progression.png` (trayectorias promedio).

---
