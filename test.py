from model import DonkeyCarMMDPEnv

exe_path = "C:\\Users\\10944\\Code\\CSCE642-Project\\DonkeySimWin\\donkey_sim.exe"
port = 9091
conf = {"exe_path": exe_path, "port": port}

# Create the environment
env = DonkeyCarMMDPEnv(conf)

# model that does nothing
for _ in range(1000):
    obs, reward, done, info = env.step([0, 0])
    if done:
        env.reset()
env.close()