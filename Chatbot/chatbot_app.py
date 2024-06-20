import streamlit as st
from streamlit.components.v1 import html

from chatbot import ConversationChainWithContext

def get_answer(user_input):
    response = ConversationChainWithContext().chat(user_input)
    st.write(response)

def main():
    st.title("GetBetterHead")
    st.title("Support Me on Buy Me a Coffee")
    buy_me_coffee_script = """
         <script type="text/javascript" src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js" data-name="bmc-button" data-slug="uladzimircd" data-color="#FFDD00" data-emoji="☕"  data-font="Poppins" data-text="Buy me a coffee" data-outline-color="#000000" data-font-color="#000000" data-coffee-color="#ffffff" ></script>
    """
    st.write(
        "If you find this app helpful and would like to support me, you can buy me a coffee!")
    st.markdown(buy_me_coffee_script, unsafe_allow_html=True)
    st.write(
        "Chat with our new mental health GetBetterHead AI assistant! This AI will utilize an extensive database consisting of the leading mental health professionals advises to answer questions and provide an invaluable feetback on user's inquiries. Please ask questions related to psychology or neurology only.")
    st.header("Disclaimer")

    disclaimer_text = """
    This web application ("GetBetterHead AI") is provided for informational purposes only. The information provided within this App is not intended to be a substitute for professional advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health providers with any questions you may have regarding a medical condition or any other professional matter.

    By using this App, you agree to the following:
    - You understand that the information provided by this App is not guaranteed to be accurate, complete, or up-to-date.
    - You assume full responsibility for any actions taken based on the information provided by this App.
    - You release the developer(s) of this App from any liability arising from your use of the App.

    This App is provided "as is" without any warranties of any kind, either express or implied, including, but not limited to, implied warranties of merchantability, fitness for a particular purpose, or non-infringement.

    The developer(s) of this App disclaim any responsibility for any harm or liability arising from the use of this App.

    If you do not agree to these terms, please do not use this App.

    """

    st.write(disclaimer_text)

    input = st.text_input("Enter your question.")

    if st.button("Get Answer"):
        get_answer(input)
