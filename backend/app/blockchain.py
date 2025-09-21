import os
from web3 import Web3
from typing import Optional
from .config import settings

class BlockchainService:
    """Blockchain integration for storing file hashes on Polygon Mumbai"""
    
    def __init__(self):
        self.rpc_url = settings.POLYGON_RPC_URL
        self.private_key = settings.POLYGON_PRIVATE_KEY
        self.contract_address = settings.CONTRACT_ADDRESS
        
        if not self.rpc_url or not self.private_key:
            raise Exception("Blockchain credentials not configured")
        
        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        
        if not self.w3.is_connected():
            raise Exception("Failed to connect to Polygon network")
        
        # Contract ABI (simplified)
        self.contract_abi = [
            {
                "inputs": [
                    {"internalType": "bytes32", "name": "_fileHash", "type": "bytes32"},
                    {"internalType": "string", "name": "_uploader", "type": "string"}
                ],
                "name": "storeHash",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"internalType": "bytes32", "name": "_fileHash", "type": "bytes32"}],
                "name": "getHashInfo",
                "outputs": [
                    {"internalType": "string", "name": "uploader", "type": "string"},
                    {"internalType": "uint256", "name": "timestamp", "type": "uint256"}
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        # Get account from private key
        self.account = self.w3.eth.account.from_key(self.private_key)
        
        # Initialize contract if address is provided
        if self.contract_address:
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=self.contract_abi
            )
        else:
            self.contract = None
    
    async def store_hash(self, file_hash: str, uploader: str) -> str:
        """Store file hash on blockchain"""
        if not self.contract:
            raise Exception("Contract not deployed or address not configured")
        
        try:
            # Convert file hash to bytes32
            file_hash_bytes = bytes.fromhex(file_hash)
            
            # Build transaction
            transaction = self.contract.functions.storeHash(
                file_hash_bytes,
                uploader
            ).build_transaction({
                'from': self.account.address,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
            })
            
            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.private_key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return receipt.transactionHash.hex()
        
        except Exception as e:
            print(f"Blockchain transaction failed: {e}")
            raise e
    
    async def get_hash_info(self, file_hash: str) -> dict:
        """Get hash information from blockchain"""
        if not self.contract:
            raise Exception("Contract not deployed or address not configured")
        
        try:
            file_hash_bytes = bytes.fromhex(file_hash)
            result = self.contract.functions.getHashInfo(file_hash_bytes).call()
            
            return {
                "uploader": result[0],
                "timestamp": result[1],
                "block_number": "N/A"  # Would need additional contract function
            }
        
        except Exception as e:
            print(f"Failed to get hash info: {e}")
            return {"error": str(e)}