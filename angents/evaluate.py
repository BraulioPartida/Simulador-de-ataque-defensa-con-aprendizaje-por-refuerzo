import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env.attacker_env import AttackerEnv
import time
import json
from tqdm import tqdm

def run_episode(model, env, deterministic=True):
    """Ejecuta un episodio completo y devuelve información detallada."""
    obs, _ = env.reset()
    done = False
    
    # Variables para seguimiento de métricas
    steps = 0
    total_reward = 0
    alert_levels = []
    data_exfiltrated = []
    nodes_compromised = []
    actions_taken = []
    resources_left = []
    episode_info = {}
    
    # Ejecutar un episodio completo
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Actualizar métricas
        steps += 1
        total_reward += reward
        alert_levels.append(info['alert_level'])
        data_exfiltrated.append(info['data_exfiltrated'])
        nodes_compromised.append(info['nodes_compromised'])
        actions_taken.append(action)
        resources_left.append(info['resources_left'])
    
    # Construir diccionario con toda la información del episodio
    episode_info = {
        'steps': steps,
        'total_reward': total_reward,
        'max_alert_level': max(alert_levels) if alert_levels else 0,
        'final_alert_level': alert_levels[-1] if alert_levels else 0,
        'total_data_exfiltrated': data_exfiltrated[-1] if data_exfiltrated else 0,
        'final_nodes_compromised': nodes_compromised[-1] if nodes_compromised else 0,
        'nodes_compromised_history': nodes_compromised,
        'alert_level_history': alert_levels,
        'data_exfiltrated_history': data_exfiltrated,
        'resources_left': resources_left[-1] if resources_left else 0,
        'action_distribution': np.bincount(actions_taken, minlength=env.action_space.n).tolist()
    }
    
    return episode_info

def evaluate_model(model, env, n_episodes=100):
    """Evalúa un modelo ejecutando múltiples episodios y recopilando estadísticas."""
    all_episodes_info = []
    
    for i in tqdm(range(n_episodes), desc=f"Evaluando modelo"):
        episode_info = run_episode(model, env, deterministic=True)
        all_episodes_info.append(episode_info)
    
    # Calcular estadísticas agregadas
    stats = {}
    
    # Métricas básicas
    metrics = ['total_reward', 'steps', 'max_alert_level', 'final_alert_level', 
               'total_data_exfiltrated', 'final_nodes_compromised', 'resources_left']
    
    for metric in metrics:
        values = [ep[metric] for ep in all_episodes_info]
        stats[f'{metric}_mean'] = float(np.mean(values))  # Convertir explícitamente a float
        stats[f'{metric}_std'] = float(np.std(values))
        stats[f'{metric}_min'] = float(np.min(values))
        stats[f'{metric}_max'] = float(np.max(values))
    
    # Tasa de éxito (definida como conseguir exfiltrar datos)
    stats['success_rate'] = float(sum(1 for ep in all_episodes_info if ep['total_data_exfiltrated'] > 0) / n_episodes)
    
    # Distribución de acciones promedio
    action_distributions = np.array([ep['action_distribution'] for ep in all_episodes_info])
    stats['action_distribution_mean'] = action_distributions.mean(axis=0).tolist()
    
    # Devolver tanto las estadísticas como los datos crudos
    return stats, all_episodes_info

def compare_models(model_paths, model_names, env_factory, n_episodes=100):
    """
    Compara múltiples modelos en un entorno dado.
    
    Args:
        model_paths: Lista de rutas a los modelos guardados
        model_names: Lista de nombres para los modelos (para visualización)
        env_factory: Función que crea una instancia del entorno
        n_episodes: Número de episodios para evaluar cada modelo
    """
    # Resultados para cada modelo
    all_results = {}
    
    for i, (path, name) in enumerate(zip(model_paths, model_names)):
        print(f"\n{'='*50}")
        print(f"Evaluando modelo {i+1}/{len(model_paths)}: {name}")
        print(f"{'='*50}")
        
        # Detectar el tipo de modelo (DQN o PPO) a partir del nombre
        if "dqn" in name.lower():
            model = DQN.load(path)
        elif "ppo" in name.lower():
            model = PPO.load(path) 
        else:
            raise ValueError(f"No se puede determinar el tipo de modelo para {name}")
        
        # Crear una nueva instancia del entorno para cada evaluación
        env = env_factory()
        
        # Evaluar el modelo
        stats, episodes_info = evaluate_model(model, env, n_episodes)
        
        # Guardar resultados
        all_results[name] = {
            'stats': stats,
            'episodes': episodes_info
        }
        
        # Liberar recursos
        env.close()
    
    # Guardar resultados en disco
    save_results(all_results, model_names)
    
    # Generar visualizaciones comparativas
    generate_comparison_visualizations(all_results, model_names)
    
    return all_results

def save_results(all_results, model_names):
    """Guarda los resultados en disco para análisis posterior."""
    os.makedirs("./evaluation_results", exist_ok=True)
    
    # Función para convertir tipos NumPy a tipos Python nativos
    def convert_numpy_types(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(i) for i in obj]
        else:
            return obj
    
    # Guardar estadísticas en formato JSON
    with open('./evaluation_results/comparison_stats.json', 'w') as f:
        # Extraer solo las estadísticas para el JSON
        stats_dict = {name: all_results[name]['stats'] for name in model_names}
        # Convertir tipos NumPy a tipos Python nativos
        stats_dict = convert_numpy_types(stats_dict)
        json.dump(stats_dict, f, indent=4)
    
    # Crear y guardar un DataFrame con los resultados para facilitar análisis
    all_episodes = []
    for model_name in model_names:
        for episode in all_results[model_name]['episodes']:
            episode_copy = convert_numpy_types(episode.copy())
            episode_copy['model'] = model_name
            all_episodes.append(episode_copy)
    
    df = pd.DataFrame(all_episodes)
    df.to_csv('./evaluation_results/all_episodes_data.csv', index=False)
    
    print("Resultados guardados en './evaluation_results/'")

def generate_comparison_visualizations(all_results, model_names):
    """Genera visualizaciones comparando los modelos."""
    os.makedirs("./evaluation_results/plots", exist_ok=True)
    
    # Configurar estilo de los gráficos
    plt.style.use('fivethirtyeight')
    sns.set_palette("Set2")
    
    # 1. Comparación de recompensas totales
    plt.figure(figsize=(12, 8))
    rewards_data = []
    for name in model_names:
        rewards = [ep['total_reward'] for ep in all_results[name]['episodes']]
        rewards_data.append(rewards)
    
    plt.boxplot(rewards_data, labels=model_names)
    plt.title('Comparativa de Recompensas Totales', fontsize=16)
    plt.ylabel('Recompensa Total', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('./evaluation_results/plots/reward_comparison.png', dpi=300, bbox_inches='tight')
    
    # 2. Comparación de métricas clave (gráfico de barras)
    metrics = ['total_reward_mean', 'steps_mean', 'final_nodes_compromised_mean', 
              'max_alert_level_mean', 'total_data_exfiltrated_mean', 'success_rate']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = [all_results[name]['stats'][metric] for name in model_names]
        ax = axes[i]
        ax.bar(model_names, values)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./evaluation_results/plots/metrics_comparison.png', dpi=300, bbox_inches='tight')
    
    # 3. Distribución de acciones
    plt.figure(figsize=(14, 8))
    bar_width = 0.35
    x = np.arange(len(all_results[model_names[0]]['stats']['action_distribution_mean']))
    
    for i, name in enumerate(model_names):
        plt.bar(x + i*bar_width, 
                all_results[name]['stats']['action_distribution_mean'], 
                width=bar_width, 
                label=name)
    
    plt.xlabel('Acción')
    plt.ylabel('Frecuencia Promedio')
    plt.title('Distribución de Acciones por Modelo')
    plt.xticks(x + bar_width/2, [f'A{i}' for i in range(len(x))])
    plt.legend()
    plt.savefig('./evaluation_results/plots/action_distribution.png', dpi=300, bbox_inches='tight')
    
    # 4. Trayectorias promedio (nodos comprometidos a lo largo del tiempo)
    plt.figure(figsize=(14, 8))
    
    for name in model_names:
        # Calcular la longitud máxima de episodio para este modelo
        max_len = max(len(ep['nodes_compromised_history']) for ep in all_results[name]['episodes'])
        
        # Inicializar matriz para almacenar todas las trayectorias
        all_trajectories = np.zeros((len(all_results[name]['episodes']), max_len))
        
        # Rellenar la matriz
        for i, ep in enumerate(all_results[name]['episodes']):
            trajectory = ep['nodes_compromised_history']
            all_trajectories[i, :len(trajectory)] = trajectory
            # El resto queda en 0
        
        # Calcular promedio por paso de tiempo
        mean_trajectory = np.mean(all_trajectories, axis=0)
        std_trajectory = np.std(all_trajectories, axis=0)
        
        # Graficar - limitar a longitud donde al menos hay datos significativos
        valid_length = np.sum(np.mean(all_trajectories > 0, axis=0) > 0.5)  # Al menos 50% de episodios tienen datos
        valid_length = max(valid_length, 20)  # Al menos mostrar 20 pasos
        
        x = np.arange(valid_length)
        plt.plot(x, mean_trajectory[:valid_length], label=name)
        plt.fill_between(x, 
                        np.maximum(0, mean_trajectory[:valid_length] - std_trajectory[:valid_length]), 
                        mean_trajectory[:valid_length] + std_trajectory[:valid_length], 
                        alpha=0.2)
    
    plt.xlabel('Paso')
    plt.ylabel('Nodos Comprometidos (Promedio)')
    plt.title('Progresión de Nodos Comprometidos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./evaluation_results/plots/nodes_progression.png', dpi=300, bbox_inches='tight')
    
    # 5. Trayectorias promedio de nivel de alerta
    plt.figure(figsize=(14, 8))
    
    for name in model_names:
        # Calcular la longitud máxima de episodio para este modelo
        max_len = max(len(ep['alert_level_history']) for ep in all_results[name]['episodes'])
        
        # Inicializar matriz para almacenar todas las trayectorias
        all_trajectories = np.zeros((len(all_results[name]['episodes']), max_len))
        
        # Rellenar la matriz
        for i, ep in enumerate(all_results[name]['episodes']):
            trajectory = ep['alert_level_history']
            all_trajectories[i, :len(trajectory)] = trajectory
            # El resto queda en 0
        
        # Calcular promedio por paso de tiempo
        mean_trajectory = np.mean(all_trajectories, axis=0)
        std_trajectory = np.std(all_trajectories, axis=0)
        
        # Graficar - limitar a longitud donde al menos hay datos significativos
        valid_length = np.sum(np.mean(all_trajectories > 0, axis=0) > 0.1)  # Al menos 10% de episodios tienen datos
        valid_length = max(valid_length, 20)  # Al menos mostrar 20 pasos
        
        x = np.arange(valid_length)
        plt.plot(x, mean_trajectory[:valid_length], label=name)
        plt.fill_between(x, 
                        np.maximum(0, mean_trajectory[:valid_length] - std_trajectory[:valid_length]), 
                        mean_trajectory[:valid_length] + std_trajectory[:valid_length], 
                        alpha=0.2)
    
    plt.xlabel('Paso')
    plt.ylabel('Nivel de Alerta (Promedio)')
    plt.title('Progresión del Nivel de Alerta')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./evaluation_results/plots/alert_progression.png', dpi=300, bbox_inches='tight')
    
    print("Visualizaciones guardadas en './evaluation_results/plots/'")

def main():
    # Configuración
    DQN_MODEL_PATH = "./models/final/dqn_attacker_final"
    PPO_MODEL_PATH = "./models/final/ppo_attacker_final"
    
    # Comprobar si existen los modelos
    models_exist = True
    for path in [DQN_MODEL_PATH, PPO_MODEL_PATH]:
        if not os.path.exists(path + ".zip"):
            print(f"⚠️ Advertencia: El modelo {path}.zip no existe.")
            models_exist = False
    
    if not models_exist:
        print("Asegúrate de haber entrenado los modelos antes de ejecutar la evaluación.")
        print("Buscando modelos alternativos...")
        
        # Buscar en carpetas de modelos para posibles alternativas
        model_dirs = ["./models/final/", "./models/best_dqn/", "./models/best_ppo/", "./models/checkpoints/"]
        dqn_models = []
        ppo_models = []
        
        for dir_path in model_dirs:
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    if file.endswith(".zip"):
                        full_path = os.path.join(dir_path, file)
                        if "dqn" in file.lower():
                            dqn_models.append(full_path)
                        elif "ppo" in file.lower():
                            ppo_models.append(full_path)
        
        # Si se encontraron modelos alternativos, usarlos
        if dqn_models and ppo_models:
            print("\nModelos alternativos encontrados:")
            print(f"DQN: {dqn_models[0]}")
            print(f"PPO: {ppo_models[0]}")
            
            DQN_MODEL_PATH = dqn_models[0].replace(".zip", "")
            PPO_MODEL_PATH = ppo_models[0].replace(".zip", "")
        else:
            print("No se encontraron modelos alternativos. Por favor, entrena los modelos primero.")
            return
    
    # Definir función para crear el entorno
    def create_env():
        return AttackerEnv(num_nodes=7, num_services=5, num_vulns=5)
    
    # Evaluar y comparar ambos modelos
    print("\n🔍 Iniciando evaluación comparativa de modelos DQN y PPO...")
    model_paths = [DQN_MODEL_PATH, PPO_MODEL_PATH]
    model_names = ["DQN", "PPO"]
    
    try:
        results = compare_models(
            model_paths=model_paths,
            model_names=model_names,
            env_factory=create_env,
            n_episodes=100  # Número de episodios para evaluación
        )
        
        # Imprimir resumen de resultados
        print("\n📊 Resumen de resultados:")
        for name in model_names:
            stats = results[name]['stats']
            print(f"\n--- {name} ---")
            print(f"Recompensa promedio: {stats['total_reward_mean']:.2f} ± {stats['total_reward_std']:.2f}")
            print(f"Pasos promedio por episodio: {stats['steps_mean']:.2f}")
            print(f"Tasa de éxito: {stats['success_rate']*100:.1f}%")
            print(f"Datos exfiltrados promedio: {stats['total_data_exfiltrated_mean']:.2f}")
            print(f"Nodos comprometidos promedio: {stats['final_nodes_compromised_mean']:.2f}")
            print(f"Nivel de alerta promedio: {stats['final_alert_level_mean']:.2f}")
        
        print("\n✅ Evaluación comparativa completada.")
        print("Se han generado gráficos y guardado resultados en la carpeta 'evaluation_results'.")
    
    except Exception as e:
        print(f"\n❌ Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()