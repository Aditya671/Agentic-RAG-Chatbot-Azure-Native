import os  
import httpx  
import jwt  
from jwt import PyJWKClient  
from fastapi import HTTPException  
from chainlit.user import User  
from chainlit.oauth_providers import OAuthProvider  
  
  
class AzureADOAuthProvider(OAuthProvider):  
    # Must match your callback path: /auth/oauth/{id}/callback  
    id = "azure-ad"  
  
    # Required environment variables  
    env = [  
        "OAUTH_AZURE_AD_CLIENT_ID",  
        "OAUTH_AZURE_AD_CLIENT_SECRET",  
        "OAUTH_AZURE_AD_TENANT_ID",  
    ]  
  
    # Build Azure AD (v2) endpoints  
    _tenant = os.environ.get("OAUTH_AZURE_AD_TENANT_ID", "common")  
    url_base = f"https://login.microsoftonline.com/{_tenant}/"  
    authorize_url = f"{url_base}oauth2/v2.0/authorize"  
    token_url = f"{url_base}oauth2/v2.0/token"  
    well_known_url = f"https://login.microsoftonline.com/{_tenant}/v2.0/.well-known/openid-configuration"  
    iss_url = f"https://login.microsoftonline.com/{_tenant}/v2.0"  
  
    def __init__(self):  
        self.client_id = os.environ.get("OAUTH_AZURE_AD_CLIENT_ID")  
        self.client_secret = os.environ.get("OAUTH_AZURE_AD_CLIENT_SECRET")  
  
        # Minimal scopes to get an id_token with email/name info  
        # Add "offline_access" if you need a refresh_token.  
        # Add Graph scopes (e.g. "https://graph.microsoft.com/User.Read") only if you plan to call Graph.  
        self.authorize_params = {  
            "response_type": "code",  
            "response_mode": "query",  
            "scope": "openid profile email offline_access",  
            # "prompt": "select_account",  # optional UX tweak  
        }  
  
    async def get_token(self, code: str, url: str) -> str:  
        """  
        Exchange authorization code for tokens and return the id_token.  
        'url' is the redirect_uri (e.g., http://localhost:8000/auth/oauth/azure-ad/callback)  
        """  
        payload = {  
            "client_id": self.client_id,  
            "client_secret": self.client_secret,  
            "code": code,  
            "grant_type": "authorization_code",  
            "redirect_uri": url,  
            "scope": "openid profile email offline_access",  
        }  
  
        async with httpx.AsyncClient() as client:  
            response = await client.post(self.token_url, data=payload)  
            try:  
                response.raise_for_status()  
            except httpx.HTTPStatusError as e:  
                raise HTTPException(status_code=400, detail=f"Token request failed: {e.response.text}")  
  
            data = response.json()  
            token = data.get("id_token")  
            if not token:  
                raise HTTPException(status_code=400, detail="No id_token in token response")  
            return token  
  
    async def get_user_info(self, token: str):  
        """  
        Validate and decode the id_token using Azure AD JWKS.  
        Return (raw_claims, Chainlit User).  
        """  
        # Fetch well-known config for JWKS and issuer  
        async with httpx.AsyncClient() as client:  
            resp = await client.get(self.well_known_url)  
            try:  
                resp.raise_for_status()  
            except httpx.HTTPStatusError as e:  
                raise HTTPException(status_code=400, detail=f"Failed to fetch well-known config: {e.response.text}")  
            well_known = resp.json()  
  
        jwks_uri = well_known.get("jwks_uri")  
        issuer = well_known.get("issuer", self.iss_url)  
        algs = well_known.get("id_token_signing_alg_values_supported", ["RS256"])  
  
        if not jwks_uri:  
            raise HTTPException(status_code=400, detail="No jwks_uri in well-known configuration")  
  
        try:  
            jwks_client = PyJWKClient(jwks_uri)  
            signing_key = jwks_client.get_signing_key_from_jwt(token).key  
  
            azure_user = jwt.decode(  
                token,  
                signing_key,  
                algorithms=algs,  
                audience=self.client_id,  
                issuer=issuer,  
                options={"verify_aud": True, "verify_iss": True},  
            )  
        except Exception as e:  
            raise HTTPException(status_code=400, detail=f"Invalid id_token: {str(e)}")  
  
        # Choose a stable identifier. Prefer email/preferred_username; fall back to oid.  
        identifier = (  
            azure_user.get("email")  
            or azure_user.get("preferred_username")  
            or azure_user.get("upn")  # rarely present in v2 tokens, but check anyway  
            or azure_user.get("oid")  
        )  
  
        if not identifier:  
            raise HTTPException(status_code=400, detail="Could not determine user identifier from token claims")  
  
        user = User(  
            identifier=azure_user["userPrincipalName"],
            metadata={
                "image": azure_user.get("image"),
                "provider": "azure-ad",
                "refresh_token": getattr(self, "_refresh_token", None),
            },
        )  
  
        return azure_user, user