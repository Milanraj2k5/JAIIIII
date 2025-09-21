from web3 import Web3
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

# Get RPC URL from .env
rpc_url = os.getenv("POLYGON_RPC_URL")
print("RPC URL from .env:", rpc_url)

# Connect
w3 = Web3(Web3.HTTPProvider(rpc_url))
print("Connected to Polygon Amoy:", w3.is_connected())
