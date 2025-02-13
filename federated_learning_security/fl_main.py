import os

print("Starting Federated Learning setup...")

os.system("python server/server.py &")
os.system("python clients/client.py &")
os.system("python clients/malicious_client.py &")
