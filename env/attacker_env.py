import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AttackerEnv(gym.Env):
    """
    Gym environment for a network attack simulation.

    State representation:
        - nodes_discovered: binary vector of length N
        - services_detected: binary matrix (N x S)
        - vulns_known: binary matrix (N x V)
        - privileges: integer vector of length N (0:none,1:user,2:root)
        - resources: scalar (tokens left)
        - nodes_compromised: binary vector of length N
        - data_exfil: scalar (normalized)

    Action space (discrete):
        0: scan(node,service)
        1: exploit(node,vuln)
        2: escalate(node)
        3: move(source_node,target_node)
        4: exfiltrate(amount)
        5: cover_tracks()
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, num_nodes=5, num_services=3, num_vulns=4, max_resources=10, max_exfil=100):
        super(AttackerEnv, self).__init__()
        self.num_nodes = num_nodes
        self.num_services = num_services
        self.num_vulns = num_vulns
        self.max_resources = max_resources
        self.max_exfil = max_exfil
        
        # Network topology - adjacency matrix
        self.network_topology = self._generate_network_topology()
        
        # Define action and observation spaces
        # Discrete actions: 6 types + parameterization internally
        self.action_space = spaces.Discrete(6)

        # Observation: will flatten all components into one vector
        obs_dim = (self.num_nodes               # nodes_discovered
                   + self.num_nodes * self.num_services  # services_detected
                   + self.num_nodes * self.num_vulns     # vulns_known
                   + self.num_nodes               # privileges
                   + 1                            # resources
                   + self.num_nodes               # nodes_compromised
                   + 1)                           # data_exfiltration
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )

        # Initialize services and vulnerabilities for the network
        self._initialize_network_properties()
        
        self.seed()
        self.reset()

    def _generate_network_topology(self):
        """Generate a random network topology (adjacency matrix)."""
        # Start with disconnected nodes
        topology = np.zeros((self.num_nodes, self.num_nodes), dtype=int)
        
        # Ensure network is connected
        for i in range(1, self.num_nodes):
            # Connect to at least one previous node
            j = self.np_random.integers(0, i) if hasattr(self, 'np_random') else np.random.randint(0, i)
            topology[i, j] = 1
            topology[j, i] = 1  # Bidirectional connection
            
        # Add some additional random connections
        for _ in range(self.num_nodes):
            i = self.np_random.integers(0, self.num_nodes) if hasattr(self, 'np_random') else np.random.randint(0, self.num_nodes)
            j = self.np_random.integers(0, self.num_nodes) if hasattr(self, 'np_random') else np.random.randint(0, self.num_nodes)
            if i != j:
                topology[i, j] = 1
                topology[j, i] = 1
                
        return topology
    
    def _initialize_network_properties(self):
        """Initialize services and vulnerabilities for each node."""
        if not hasattr(self, 'np_random'):
            self.np_random = np.random.RandomState()
            
        # Each node has some services running
        self.node_services = np.zeros((self.num_nodes, self.num_services), dtype=int)
        for node in range(self.num_nodes):
            num_services = self.np_random.integers(1, self.num_services + 1)
            service_indices = self.np_random.choice(self.num_services, num_services, replace=False)
            self.node_services[node, service_indices] = 1
            
        # Each service may have vulnerabilities
        self.service_vulns = np.zeros((self.num_nodes, self.num_services, self.num_vulns), dtype=int)
        for node in range(self.num_nodes):
            for service in range(self.num_services):
                if self.node_services[node, service]:
                    # Each active service has a chance of having vulnerabilities
                    for vuln in range(self.num_vulns):
                        if self.np_random.random() < 0.3:  # 30% chance of vulnerability
                            self.service_vulns[node, service, vuln] = 1
        
        # Data value per node (how much valuable data can be exfiltrated)
        self.node_data_value = np.zeros(self.num_nodes)
        for node in range(self.num_nodes):
            self.node_data_value[node] = self.np_random.random() * self.max_exfil

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
            # Re-initialize network with new seed
            self.network_topology = self._generate_network_topology()
            self._initialize_network_properties()

        # Start with one node discovered (entry point)
        self.nodes_discovered = np.zeros(self.num_nodes, dtype=int)
        self.nodes_discovered[0] = 1  # Start with node 0 discovered
        
        self.services_detected = np.zeros((self.num_nodes, self.num_services), dtype=int)
        self.vulns_known = np.zeros((self.num_nodes, self.num_vulns), dtype=int)
        self.privileges = np.zeros(self.num_nodes, dtype=int)
        self.resources = self.max_resources
        self.nodes_compromised = np.zeros(self.num_nodes, dtype=int)
        self.data_exfil = 0.0
        self.steps = 0
        self.alert_level = 0.0  # Track detection risk
        
        # Current position in the network
        self.current_node = 0

        return self._get_observation(), {}

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # Interpret action
        # 0: scan, 1: exploit, 2: escalate, 3: move, 4: exfiltrate, 5: cover tracks
        if action == 0:  # scan
            # Scan a random node that's been discovered
            discovered_nodes = np.where(self.nodes_discovered == 1)[0]
            if len(discovered_nodes) > 0:
                node = self.np_random.choice(discovered_nodes)
                service = self.np_random.integers(0, self.num_services)
                
                # Only discover services that actually exist
                if self.node_services[node, service] == 1:
                    self.services_detected[node, service] = 1
                    
                    # Discover vulnerabilities with some probability
                    for vuln in range(self.num_vulns):
                        if self.service_vulns[node, service, vuln] == 1 and self.np_random.random() < 0.7:
                            self.vulns_known[node, vuln] = 1
                            reward += 0.5  # Bonus for finding a vulnerability
                    
                    reward += 0.05
                else:
                    reward -= 0.05  # Penalty for scanning non-existent service
                
                # Small chance of increasing alert level
                if self.np_random.random() < 0.05:
                    self.alert_level += 0.05
            else:
                reward -= 0.1  # Cannot scan if no nodes are discovered
                
            self.resources -= 1
                
        elif action == 1:  # exploit
            # Try to exploit a vulnerability on a discovered node
            discovered_nodes = np.where(self.nodes_discovered == 1)[0]
            if len(discovered_nodes) > 0:
                node = self.np_random.choice(discovered_nodes)
                
                # Check if there are any known vulnerabilities for this node
                if np.sum(self.vulns_known[node]) > 0:
                    # Choose a random known vulnerability
                    known_vulns = np.where(self.vulns_known[node] == 1)[0]
                    vuln = self.np_random.choice(known_vulns)
                    
                    # Attempt to exploit
                    success = self.np_random.random() < 0.7  # 70% success rate
                    if success:
                        self.nodes_compromised[node] = 1
                        self.privileges[node] = 1  # Get user privileges
                        reward += 1.2
                        
                        # Discover connected nodes
                        for i in range(self.num_nodes):
                            if self.network_topology[node, i] == 1:
                                self.nodes_discovered[i] = 1
                    else:
                        reward -= 0.2
                        # Failed exploit increases alert level
                        self.alert_level += 0.1
                else:
                    reward -= 0.1  # No known vulnerabilities
            else:
                reward -= 0.1  # No nodes to exploit
                
            self.resources -= 1
            
        elif action == 2:  # escalate privileges
            # Try to escalate privileges on a compromised node
            compromised_nodes = np.where(self.nodes_compromised == 1)[0]
            if len(compromised_nodes) > 0:
                node = self.np_random.choice(compromised_nodes)
                
                # Can only escalate if currently at user level
                if self.privileges[node] == 1:
                    success = self.np_random.random() < 0.5  # 50% chance to escalate
                    if success:
                        self.privileges[node] = 2  # Escalate to root
                        reward += 2.0
                    else:
                        reward -= 0.3
                        # Failed escalation significantly increases alert level
                        self.alert_level += 0.15
                else:
                    reward -= 0.1  # Already root or no privileges
            else:
                reward -= 0.1  # No compromised nodes
                
            self.resources -= 1
            
        elif action == 3:  # move between nodes
            # Move from current node to a connected compromised node
            if np.sum(self.nodes_compromised) > 1:  # Need at least 2 compromised nodes
                possible_targets = []
                for i in range(self.num_nodes):
                    # Can move if target is compromised and connected to current
                    if (i != self.current_node and 
                        self.nodes_compromised[i] == 1 and 
                        self.network_topology[self.current_node, i] == 1):
                        possible_targets.append(i)
                
                if possible_targets:
                    target = self.np_random.choice(possible_targets)
                    self.current_node = target
                    reward += 0.1
                else:
                    reward -= 0.1  # No valid targets
            else:
                reward -= 0.1  # Not enough compromised nodes
                
            self.resources -= 0.5  # Moving costs less resources
            
        elif action == 4:  # exfiltrate data
            # Exfiltrate data from the current node if compromised
            if self.nodes_compromised[self.current_node] == 1:
                # Amount depends on privilege level (root gets more)
                privilege_multiplier = 1.0 if self.privileges[self.current_node] == 1 else 2.0
                
                # Extract an amount based on node's value
                remaining_data = self.node_data_value[self.current_node] - (self.data_exfil * self.max_exfil)
                if remaining_data > 0:
                    # Extract a portion of remaining data
                    amount = min(remaining_data, self.np_random.random() * 0.3 * remaining_data * privilege_multiplier)
                    self.data_exfil += amount / self.max_exfil  # Normalize to 0-1
                    reward += amount / 10.0  # Reward proportional to data value
                    
                    # Exfiltration has high risk of detection
                    self.alert_level += 0.1 * amount / self.max_exfil
                else:
                    reward -= 0.05  # No more data to exfiltrate
            else:
                reward -= 0.1  # Cannot exfiltrate from non-compromised node
                
            self.resources -= 1
            
        elif action == 5:  # cover tracks
            # Reduce alert level at the cost of resources
            reduction = 0.3  # Can reduce alert level by up to 30%
            self.alert_level = max(0, self.alert_level - reduction)
            reward += 0.5  # Reward for reducing detection risk
            self.resources -= 2  # Covering tracks costs more resources
        
        # Additional termination conditions
        if self.data_exfil >= 0.8:  # Achieved significant data exfiltration
            reward += 5.0  # Big bonus for reaching goal
            terminated = True
            info['success'] = True
            
        if self.alert_level >= 1.0:  # Detected!
            reward -= 5.0  # Big penalty for being detected
            terminated = True
            info['detected'] = True
            
        self.steps += 1
        
        # Termination conditions
        if self.resources <= 0 or self.steps >= 100:
            terminated = True
            
        # Additional info
        info['alert_level'] = self.alert_level
        info['data_exfiltrated'] = self.data_exfil
        info['nodes_compromised'] = np.sum(self.nodes_compromised)
        info['resources_left'] = self.resources

        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        # Flatten all state components into one vector
        components = [
            self.nodes_discovered.flatten(),
            self.services_detected.flatten(),
            self.vulns_known.flatten(),
            self.privileges.flatten() / 2.0,  # Normalize privileges (0,1,2) -> (0,0.5,1)
            np.array([self.resources / self.max_resources]),
            self.nodes_compromised.flatten(),
            np.array([self.data_exfil]),
        ]
        obs = np.concatenate(components).astype(np.float32)
        return obs

    def render(self, mode='human'):
        # Simple logs of current state
        print(f"Step: {self.steps}")
        print(f"Current Node: {self.current_node}")
        print(f"Discovered: {self.nodes_discovered}")
        print(f"Compromised: {self.nodes_compromised}")
        print(f"Privileges: {self.privileges}")
        print(f"Resources: {self.resources}")
        print(f"Alert Level: {self.alert_level:.2f}")
        print(f"Data Exfiltrated: {self.data_exfil:.2f} ({self.data_exfil * self.max_exfil:.1f} units)")

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

# Example usage:
if __name__ == "__main__":
    env = AttackerEnv(num_nodes=5, num_services=3, num_vulns=4)
    obs, _ = env.reset()
    
    print("Initial state:")
    env.render()
    
    for i in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nAction: {action}")
        print(f"Reward: {reward:.2f}")
        env.render()
        
        if terminated:
            print("Environment terminated!")
            break