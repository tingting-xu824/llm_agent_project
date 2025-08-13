"""
Azure Storage service for file uploads
"""
import os
import uuid
from datetime import datetime, timedelta
from typing import Optional
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.core.exceptions import AzureError

class AzureStorageService:
    """Azure Blob Storage service for file uploads"""
    
    def __init__(self):
        # Azure Storage configuration from environment variables
        self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = os.getenv("AZURE_STORAGE_CONTAINER", "finalreports")
        self.account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self.account_key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
        
        # Validate required environment variables
        if not self.connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is required")
        if not self.account_name:
            raise ValueError("AZURE_STORAGE_ACCOUNT_NAME environment variable is required")
        if not self.account_key:
            raise ValueError("AZURE_STORAGE_ACCOUNT_KEY environment variable is required")
        
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            # Ensure container exists
            self._ensure_container_exists()
        except Exception as e:
            print(f"Error initializing Azure Storage: {e}")
            raise
    
    def _ensure_container_exists(self):
        """Ensure the container exists, create if it doesn't"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            container_client.get_container_properties()
        except AzureError:
            # Container doesn't exist, create it
            self.blob_service_client.create_container(self.container_name)
            print(f"Created container: {self.container_name}")
    
    def upload_file(self, file_content: bytes, file_name: str, user_id: int) -> Optional[str]:
        """
        Upload file to Azure Blob Storage
        
        Args:
            file_content: File content as bytes
            file_name: Original file name
            user_id: User ID for organizing files
            
        Returns:
            str: Public URL of the uploaded file, or None if upload failed
        """
        try:
            # Generate unique blob name
            file_extension = os.path.splitext(file_name)[1]
            unique_id = str(uuid.uuid4())
            blob_name = f"user_{user_id}/{unique_id}{file_extension}"
            
            # Get blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=blob_name
            )
            
            # Upload file
            blob_client.upload_blob(file_content, overwrite=True)
            
            # Generate public URL
            file_url = self._generate_public_url(blob_name)
            
            print(f"File uploaded successfully: {blob_name}")
            return file_url
            
        except Exception as e:
            print(f"Error uploading file to Azure Storage: {e}")
            return None
    
    def _generate_public_url(self, blob_name: str) -> str:
        """
        Generate public URL for the blob
        
        Args:
            blob_name: Name of the blob in storage
            
        Returns:
            str: Public URL with SAS token
        """
        # Generate SAS token for public access (read-only)
        sas_token = generate_blob_sas(
            account_name=self.account_name,
            container_name=self.container_name,
            blob_name=blob_name,
            account_key=self.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(days=365)  # 1 year expiry
        )
        
        # Construct public URL
        public_url = f"https://{self.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}?{sas_token}"
        return public_url
    
    def delete_file(self, blob_name: str) -> bool:
        """
        Delete file from Azure Blob Storage
        
        Args:
            blob_name: Name of the blob to delete
            
        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=blob_name
            )
            blob_client.delete_blob()
            print(f"File deleted successfully: {blob_name}")
            return True
        except Exception as e:
            print(f"Error deleting file from Azure Storage: {e}")
            return False

# Global instance
azure_storage = AzureStorageService()
