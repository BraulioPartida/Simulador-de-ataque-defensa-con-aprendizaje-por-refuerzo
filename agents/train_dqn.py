from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from env.attacker_env import AttackerEnv

def main():
    # Crear el directorio para los logs y modelos
    os.makedirs("./logs/eval/", exist_ok=True)
    os.makedirs("./models/best_dqn/", exist_ok=True)
    os.makedirs("./models/checkpoints/", exist_ok=True)
    os.makedirs("./models/final/", exist_ok=True)
    
    # Parámetros de configuración
    TRAIN_TIMESTEPS = 500_000  # Más pasos para mejor convergencia
    LEARNING_RATE = 3e-4       # Ajustado para mejor equilibrio exploración/explotación
    BATCH_SIZE = 128           # Aumentado para reducir varianza en las actualizaciones
    SEED = 42
    
    # Crear entorno
    env = AttackerEnv(num_nodes=7, num_services=5, num_vulns=5)  # Entorno más complejo
    
    # Verificación básica del entorno (opcional, pero útil para debugging)
    try:
        check_env(env, warn=True)
        print("✅ Environment check passed!")
    except Exception as e:
        print(f"⚠️ Environment check failed: {e}")
        return
    
    # Wrapping del entorno en VecEnv para mejor rendimiento y consistencia
    # (incluso con un solo entorno, esto permite escalabilidad futura)
    def make_env():
        def _init():
            env = AttackerEnv(num_nodes=7, num_services=5, num_vulns=5)
            env.seed(SEED)
            return env
        return _init
    
    # Monitor para logging
    env = DummyVecEnv([make_env()])
    env = VecMonitor(env, filename="./logs/monitor")
    
    # Entorno para evaluación
    eval_env = DummyVecEnv([lambda: Monitor(AttackerEnv(num_nodes=7, num_services=5, num_vulns=5))])
    
    # Callbacks
    # Evaluación periódica
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_dqn/",
        log_path="./logs/eval/",
        eval_freq=10000,  # Evaluar cada 10k pasos
        n_eval_episodes=10,  # Usar 10 episodios para evaluación
        deterministic=True,
        render=False,
    )
    
    # Guardar checkpoints periódicamente
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # Guardar cada 50k pasos
        save_path="./models/checkpoints/",
        name_prefix="dqn_attacker",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    # Crear modelo DQN con hiperparámetros mejorados
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        buffer_size=200_000,    # Buffer más grande para mejor aprendizaje
        batch_size=BATCH_SIZE,
        gamma=0.99,             # Factor de descuento estándar
        train_freq=4,           # Frecuencia de actualización
        gradient_steps=1,       # Pasos de gradiente por actualización
        target_update_interval=1500,  # Actualización del objetivo más frecuente
        exploration_fraction=0.1,  # Exploración más larga
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,       # Recorte de gradiente para estabilidad
        verbose=1,
        tensorboard_log="./logs/dqn_attacker/",
        seed=SEED,
        device="auto"           # Usar GPU si está disponible
    )
    
    # Entrenar el modelo
    print("🚀 Starting training...")
    start_time = time.time()
    model.learn(
        total_timesteps=TRAIN_TIMESTEPS, 
        callback=[eval_callback, checkpoint_callback],
        log_interval=100,
        tb_log_name=f"dqn_attacker_lr{LEARNING_RATE}"
    )
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Guardar el modelo final
    model.save("./models/final/dqn_attacker_final")
    print("💾 Final model saved")
    
    # Evaluar el modelo entrenado
    print("📊 Evaluating final model...")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"🏆 Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Ejecutar algunas pruebas y visualizar el comportamiento
    test_env = AttackerEnv(num_nodes=7, num_services=5, num_vulns=5)
    obs, _ = test_env.reset()
    
    # Metricas para seguimiento
    rewards = []
    alert_levels = []
    data_exfil = []
    nodes_compromised = []
    
    # Ejecutar un episodio
    done = False
    total_reward = 0
    print("\n🕹️ Running a test episode...")
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Registrar métricas
        rewards.append(reward)
        alert_levels.append(info['alert_level'])
        data_exfil.append(info['data_exfiltrated'])
        nodes_compromised.append(info['nodes_compromised'])
        
        # Mostrar información relevante
        print(f"Step: {len(rewards)}, Action: {action}, Reward: {reward:.2f}")
        print(f"Alert level: {info['alert_level']:.2f}, Data exfiltrated: {info['data_exfiltrated']:.2f}")
        print(f"Nodes compromised: {info['nodes_compromised']}/{test_env.num_nodes}, Resources: {info['resources_left']}")
        print("---")
    
    print(f"Episode finished with total reward: {total_reward:.2f}")
    
    # Visualizar resultados
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(rewards, 'b-')
    plt.title('Rewards per Step')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    
    plt.subplot(2, 2, 2)
    plt.plot(alert_levels, 'r-')
    plt.title('Alert Level')
    plt.xlabel('Steps')
    plt.ylabel('Alert Level')
    
    plt.subplot(2, 2, 3)
    plt.plot(data_exfil, 'g-')
    plt.title('Data Exfiltrated')
    plt.xlabel('Steps')
    plt.ylabel('Data Exfiltrated')
    
    plt.subplot(2, 2, 4)
    plt.plot(nodes_compromised, 'y-')
    plt.title('Nodes Compromised')
    plt.xlabel('Steps')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('./logs/dqn_attacker_results.png')
    plt.show()


if __name__ == "__main__":
    main()