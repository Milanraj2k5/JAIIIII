import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from web3 import Web3
from typing import Optional
from .config import settings


# ----------------------------
# Blockchain Service
# ----------------------------
class BlockchainService:
    """Blockchain integration for storing file hashes on Polygon Amoy"""

    def __init__(self):
        self.rpc_url = settings.POLYGON_RPC_URL
        self.private_key = settings.POLYGON_PRIVATE_KEY
        self.contract_address = settings.CONTRACT_ADDRESS

        if not self.rpc_url or not self.private_key:
            print("⚠️ Blockchain credentials not configured in .env")
            self.connected = False
            self.w3 = None
            self.account = None
            self.contract = None
            return

        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

        if not self.w3.is_connected():
            print("⚠️ Failed to connect to Polygon RPC. Blockchain features disabled.")
            self.connected = False
            self.account = None
            self.contract = None
            return

        # Load account
        self.account = self.w3.eth.account.from_key(self.private_key)
        print(f"✅ Connected to Polygon as {self.account.address}")
        self.connected = True

        # Contract ABI
        self.contract_abi = [
            {
                "inputs": [
                    {"internalType": "bytes32", "name": "_fileHash", "type": "bytes32"},
                    {"internalType": "string", "name": "_uploader", "type": "string"},
                ],
                "name": "storeHash",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function",
            },
            {
                "inputs": [{"internalType": "bytes32", "name": "_fileHash", "type": "bytes32"}],
                "name": "getHashInfo",
                "outputs": [
                    {"internalType": "string", "name": "uploader", "type": "string"},
                    {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                ],
                "stateMutability": "view",
                "type": "function",
            },
        ]

        # Attach contract
        if self.contract_address:
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=self.contract_abi,
            )
        else:
            print("⚠️ Contract address not set. Only wallet is connected.")
            self.contract = None

    async def store_hash(self, file_hash: str, uploader: str) -> str:
        """Store file hash on blockchain"""
        if not self.connected or not self.contract:
            raise Exception("Blockchain not connected or contract missing")

        try:
            # Validate hash
            if not file_hash.startswith("0x"):
                file_hash = "0x" + file_hash
            file_hash_bytes = self.w3.to_bytes(hexstr=file_hash)

            # Build transaction
            transaction = self.contract.functions.storeHash(
                file_hash_bytes, uploader
            ).build_transaction({
                "from": self.account.address,
                "gas": 200000,
                "gasPrice": self.w3.eth.gas_price,
                "nonce": self.w3.eth.get_transaction_count(self.account.address),
            })

            # Sign and send
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)

            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            return receipt.transactionHash.hex()

        except Exception as e:
            print(f"❌ Blockchain transaction failed: {e}")
            raise e

    async def get_hash_info(self, file_hash: str) -> dict:
        """Get hash information from blockchain"""
        if not self.connected or not self.contract:
            raise Exception("Blockchain not connected or contract missing")

        try:
            if not file_hash.startswith("0x"):
                file_hash = "0x" + file_hash
            file_hash_bytes = self.w3.to_bytes(hexstr=file_hash)

            result = self.contract.functions.getHashInfo(file_hash_bytes).call()

            return {
                "uploader": result[0],
                "timestamp": result[1],
            }

        except Exception as e:
            print(f"❌ Failed to get hash info: {e}")
            return {"error": str(e)}


# ----------------------------
# FastAPI Router
# ----------------------------
router = APIRouter()
blockchain_service = BlockchainService()


class StoreHashRequest(BaseModel):
    file_hash: str
    uploader: str


@router.post("/store-hash")
async def store_hash(request: StoreHashRequest):
    try:
        tx_hash = await blockchain_service.store_hash(request.file_hash, request.uploader)
        return {"status": "success", "tx_hash": tx_hash}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/get-hash-info/{file_hash}")
async def get_hash_info(file_hash: str):
    try:
        info = await blockchain_service.get_hash_info(file_hash)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# NEW: Blockchain Status Endpoint
# ----------------------------
@router.get("/blockchain-status")
async def blockchain_status():
    """Check blockchain connection status"""
    try:
        return {
            "connected": blockchain_service.connected,
            "account": blockchain_service.account.address if blockchain_service.account else None,
            "contract_loaded": blockchain_service.contract is not None
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}
