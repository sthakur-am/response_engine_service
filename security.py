import logging

from msal import PublicClientApplication

app = PublicClientApplication(
    "CLIENT_ID",
    # "credential",
    authority = "https://login.microsoftonline.com/dd5e230f-c165-49c4-957f-e203458fffab",
    # redirectUri = "https://localhost:3000/taskpane.html"
)

def get_user_token():
    result = app.acquire_token_interactive(scopes=["User.Read"])

    if "access_token" in result:
        return result["acess_token"]
    else:
        logging.log(result.get("error"))
        logging.log(result.get("error_description"))
        logging.log(result.get("correlation_id"))
        raise RuntimeError(result.get("error"))



# get_user_token()
