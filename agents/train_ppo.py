from stable_baselines3 import PPO
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
    os.makedirs("./models/best_ppo/", exist_ok=True)
    os.makedirs("./models/checkpoints/", exist_ok=True)
    os.makedirs("./models/final/", exist_ok=True)
    
    # Parámetros de configuración
    TRAIN_TIMESTEPS = 1_000_000  # Más pasos que DQN para mejor convergencia
    LEARNING_RATE = 3e-4        # Tasa de aprendizaje adaptada para PPO
    BATCH_SIZE = 64             # Tamaño de lote para muestreo
    N_STEPS = 1024               # Pasos por actualización (horizonte)
    SEED = 42
    
    # Crear entorno
    env = AttackerEnv(num_nodes=7, num_services=5, num_vulns=5)
    
    # Verificación básica del entorno
    try:
        check_env(env, warn=True)
        print("✅ Environment check passed!")
    except Exception as e:
        print(f"⚠️ Environment check failed: {e}")
        return
    
    # Wrapping del entorno en VecEnv para paralelización
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
        best_model_save_path="./models/best_ppo/",
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
        name_prefix="ppo_attacker",
        save_replay_buffer=False,  # PPO no usa buffer de experiencia como DQN
        save_vecnormalize=True,
    )
    
    # Crear modelo PPO con hiperparámetros optimizados
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,            # Horizonte para actualización
        batch_size=BATCH_SIZE,
        n_epochs=10,                # Número de epochs por actualización
        gamma=0.99,                 # Factor de descuento
        gae_lambda=0.95,            # Parámetro de estimación de ventaja generalizada
        clip_range=0.3,             # Parámetro de recorte para PPO
        clip_range_vf=None,         # No recortar función de valor
        normalize_advantage=True,   # Normalizar ventajas
        ent_coef=0.1,              # Coeficiente de entropía para exploración
        vf_coef=0.5,                # Coeficiente de función de valor
        max_grad_norm=0.5,          # Recorte de gradiente para estabilidad
        use_sde=False,              # No usar exploración dependiente del estado
        sde_sample_freq=-1,
        target_kl=None,             # No usar early stopping basado en KL divergence
        tensorboard_log="./logs/ppo_attacker/",
        verbose=1,
        seed=SEED,
        device="auto"               # Usar GPU si está disponible
    )
    
    # Entrenar el modelo
    print("🚀 Starting PPO training...")
    start_time = time.time()
    model.learn(
        total_timesteps=TRAIN_TIMESTEPS, 
        callback=[eval_callback, checkpoint_callback],
        log_interval=10,
        tb_log_name=f"ppo_attacker_lr{LEARNING_RATE}"
    )
    training_time = time.time() - start_time
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Guardar el modelo final
    model.save("./models/final/ppo_attacker_final")
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
    plt.savefig('./logs/ppo_attacker_results.png')
    plt.show()


if __name__ == "__main__":
    main()