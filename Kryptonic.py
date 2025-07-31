import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd
import google.generativeai as genai
import re

# Page configuration
st.set_page_config(
    page_title="Kryptonic AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration - ADD YOUR GEMINI API KEY HERE
# IMPORTANT: For security, never hardcode API keys in production apps.
# Use Streamlit secrets or environment variables.
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # Replace with your actual key or use st.secrets
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

class CryptoChatbot:
    def __init__(self, gemini_api_key):  # Fixed: was _init_
        # Configure Gemini API
        genai.configure(api_key=gemini_api_key)
        # Updated model name - try the latest available model
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            st.error(f"Failed to load gemini-1.5-flash: {e}. Trying gemini-1.0-pro...")
            try:
                self.model = genai.GenerativeModel('gemini-1.0-pro')
            except Exception as e:
                st.error(f"Failed to load gemini-1.0-pro: {e}. Trying models/gemini-pro...")
                try:
                    self.model = genai.GenerativeModel('models/gemini-pro')
                except Exception as e:
                    st.error(f"Failed to load models/gemini-pro: {e}. AI chatbot will not function.")
                    self.model = None # Set model to None if all fail
        
        self.supported_coins = [
            'bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana',
            'polkadot', 'dogecoin', 'avalanche-2', 'chainlink', 'polygon',
            'ripple', 'litecoin', 'stellar', 'monero', 'tron'
        ]
        
        # Crypto-related keywords for filtering
        self.crypto_keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'cryptocurrency', 'crypto', 'blockchain',
            'altcoin', 'defi', 'nft', 'token', 'coin', 'mining', 'wallet', 'exchange',
            'trading', 'hodl', 'market cap', 'volume', 'price', 'bullish', 'bearish',
            'satoshi', 'wei', 'gas', 'fees', 'staking', 'yield', 'liquidity', 'dapp',
            'smart contract', 'consensus', 'proof of work', 'proof of stake', 'fork',
            'halving', 'airdrop', 'ico', 'ido', 'dao', 'web3', 'metaverse',
            'cardano', 'ada', 'solana', 'sol', 'polkadot', 'dot', 'chainlink', 'link',
            'polygon', 'matic', 'avalanche', 'avax', 'dogecoin', 'doge', 'shiba',
            'usdt', 'usdc', 'busd', 'stable', 'tether', 'binance', 'coinbase',
            'bull market', 'bear market', 'moon', 'lambo', 'diamond hands', 'paper hands'
        ]
    
    def is_crypto_related(self, text):
        """Check if the text is cryptocurrency related"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.crypto_keywords)
    
    def get_crypto_price(self, coin_id):
        """Get current price and basic info for a cryptocurrency"""
        try:
            url = f"{COINGECKO_BASE_URL}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true'
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Error fetching crypto price for {coin_id}: {e}")
            return None
    
    def get_trending_coins(self):
        """Get trending cryptocurrencies"""
        try:
            url = f"{COINGECKO_BASE_URL}/search/trending"
            response = requests.get(url)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Error fetching trending coins: {e}")
            return None
    
    def get_market_overview(self):
        """Get top cryptocurrencies by market cap"""
        try:
            url = f"{COINGECKO_BASE_URL}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 10,
                'page': 1,
                'sparkline': 'false'
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Error fetching market overview: {e}")
            return None
    
    def get_current_market_data(self, num_top_coins=5, num_trending_coins=3):
        """Get current market data to provide context to AI"""
        market_data = self.get_market_overview()
        trending_data = self.get_trending_coins()
        
        context = "Current Crypto Market Data:\n"
        
        if market_data:
            context += f"Top {num_top_coins} Cryptocurrencies by Market Cap:\n"
            for i, coin in enumerate(market_data[:num_top_coins], 1):
                price = coin['current_price']
                change = coin['price_change_percentage_24h']
                context += f"{i}. {coin['name']} ({coin['symbol'].upper()}): ${price:,.2f} ({change:+.2f}%)\n"
        
        if trending_data and 'coins' in trending_data:
            context += f"\nTrending Coins (Top {num_trending_coins}):\n"
            for i, coin in enumerate(trending_data['coins'][:num_trending_coins], 1):
                context += f"{i}. {coin['item']['name']} ({coin['item']['symbol']})\n"
        
        return context
    
    def ask_ai(self, user_question):
        """Ask Gemini AI about cryptocurrency topics only"""
        
        if not self.model:
            return "AI model not loaded. Please check API key and internet connection."

        # First check if the question is crypto-related
        if not self.is_crypto_related(user_question):
            return "🚫 Hey! I'm only here to chat about crypto stuff - Bitcoin, Ethereum, NFTs, all that good stuff. Hit me with a crypto question! 😄"
        
        try:
            # Get current market data for context
            market_context = self.get_current_market_data()
            
            # Create prompt for Gemini
            prompt = f"""You are a professional, teen-friendly cryptocurrency assistant. You're talking to young people (ages 17-30) who want to learn about crypto. Keep the statistics local like for the Caribbean

PERSONALITY:
- Be friendly, enthusiastic, and relatable
- Use simple, everyday language (no fancy financial jargon)
- Be like a knowledgeable friend explaining crypto
- Use emojis but don't overdo it
- Keep it real and honest about risks
- Make complex topics easy to understand

RULES:
1. ONLY answer crypto-related questions
2. If it's not about crypto, redirect nicely to crypto topics
3. Explain things simply but accurately and professionally 
4. Use current market data when it helps
5. Keep responses under 100 words
6. Be encouraging but realistic about crypto investing
7. Always mention that crypto is risky
8. Answer in point form where necessary.
9. **IMPORTANT: DO NOT include any HTML tags or markdown that creates HTML elements in your response, such as <div>, <span>, etc.**

TONE EXAMPLES:
- Instead of "utilize" say "use"
- Instead of "substantial" say "big" or "huge"
- Instead of "fluctuations" say "price changes"
- Instead of "portfolio diversification" say "spreading your money around"

Current Market Context:
{market_context}

Your Question: {user_question}

Give a helpful, friendly and professional response:"""

            response = self.model.generate_content(prompt)
            
            # Strip any remaining HTML tags from the response before returning
            clean_response = re.sub(r'<.*?>', '', response.text)
            return clean_response
            
        except Exception as e:
            return f"Oops! Something went wrong on my end 😅 Try asking again in a second: {str(e)}"
    
    def process_query(self, user_input):
        """Process user query and return appropriate response"""
        user_input_lower = user_input.lower()
        
        # Check for specific data requests first
        if "price" in user_input_lower and any(coin.replace('-', '').replace('2', '') in user_input_lower.replace(' ', '') for coin in self.supported_coins):
            for coin in self.supported_coins:
                if coin.replace('-', '').replace('2', '') in user_input_lower.replace(' ', ''):
                    return self.handle_price_query(coin)
        
        elif "trending" in user_input_lower or "popular" in user_input_lower:
            return self.handle_trending_query()
        
        elif "market" in user_input_lower and ("overview" in user_input_lower or "top" in user_input_lower):
            return self.handle_market_query()
        
        # For all other questions, use AI
        else:
            return self.ask_ai(user_input)
    
    def handle_price_query(self, coin_id):
        """Handle price-related queries"""
        data = self.get_crypto_price(coin_id)
        if data and coin_id in data:
            coin_data = data[coin_id]
            price = coin_data['usd']
            change_24h = coin_data.get('usd_24h_change', 0)
            market_cap = coin_data.get('usd_market_cap', 0)
            volume_24h = coin_data.get('usd_24h_vol', 0)
            
            change_emoji = "📈" if change_24h > 0 else "📉"
            change_text = "going up" if change_24h > 0 else "going down"
            
            response = f"""**{coin_id.replace('-', ' ').title()} Right Now** 💰

💵 **Price:** ${price:,.2f}
{change_emoji} **24h:** {change_24h:+.2f}% ({change_text})
📊 **Market Size:** ${market_cap:,.0f}
💹 **Daily Trading:** ${volume_24h:,.0f}

Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC

Remember: Crypto prices change super fast! ⚡"""
            return response
        else:
            return f"Hmm, couldn't grab the price for {coin_id} right now 🤔 Maybe try again in a bit?"
    
    def handle_trending_query(self):
        """Handle trending coins query"""
        trending_data = self.get_trending_coins()
        if trending_data:
            trending_coins = trending_data['coins'][:5]
            response = "🔥 **What's Hot Right Now:**\n\n"
            for i, coin in enumerate(trending_coins, 1):
                response += f"{i}. **{coin['item']['name']}** ({coin['item']['symbol'].upper()})\n"
                response += f"    Market Rank: #{coin['item']['market_cap_rank']}\n\n"
            response += "These are the coins everyone's talking about today! 🚀"
            return response
        else:
            return "Can't get the trending list right now 😕 Try again in a moment!"
    
    def handle_market_query(self):
        """Handle market overview query"""
        market_data = self.get_market_overview()
        if market_data:
            # Create a simplified table view
            response = "📊 **Top 10 Biggest Cryptos Right Now:**\n\n"
            for i, coin in enumerate(market_data[:10], 1):
                price = coin['current_price']
                change = coin['price_change_percentage_24h']
                market_cap = coin['market_cap']
                change_emoji = "📈" if change > 0 else "📉"
                
                response += f"{i}. **{coin['name']}** ({coin['symbol'].upper()})\n"
                response += f"   💰 ${price:,.2f} {change_emoji} {change:+.2f}%\n"
                response += f"   📊 Market Cap: ${market_cap:,.0f}\n\n"
            
            response += "*These are ranked by how much they're worth in total! 💎*"
            return response
        else:
            return "Oops! Can't load the market data right now 📊 Give it another shot!"

def get_theme_css(is_dark_mode, animations_enabled):
    """Generate CSS based on theme and animation preferences"""
    
    # Base colors for themes
    if is_dark_mode:
        bg_color = "#0a0a0a"
        primary_color = "#00ff88"
        secondary_color = "#00d4ff"
        accent_color = "#ff0080"
        text_color = "#ffffff"
        surface_color = "rgba(255, 255, 255, 0.05)"
        border_color = "rgba(0, 255, 136, 0.3)"
    else:
        bg_color = "#f8fafc"
        primary_color = "#059669"
        secondary_color = "#0284c7"
        accent_color = "#dc2626"
        text_color = "#1f2937"
        surface_color = "rgba(255, 255, 255, 0.8)"
        border_color = "rgba(5, 150, 105, 0.3)"
    
    # Animation styles (conditional)
    animation_css = ""
    if animations_enabled:
        animation_css = """
        @keyframes gradientShift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .main-header {
            animation: gradientShift 3s ease infinite;
        }
        
        .glow-text {
            animation: pulse 2s infinite;
        }
        """
    
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    .stApp {{
        background: linear-gradient(135deg, {bg_color} 0%, {primary_color}20 100%);
        color: {text_color};
    }}
    
    .main-header {{
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, {primary_color}, {secondary_color}, {accent_color});
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Orbitron', monospace;
        font-weight: 900;
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }}
    
    .sub-header {{
        text-align: center;
        color: {text_color};
        opacity: 0.8;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }}
    
    .stButton > button {{
        background: linear-gradient(45deg, {primary_color}, {secondary_color});
        color: {text_color if not is_dark_mode else '#000'};
        border: none;
        border-radius: 8px;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.7rem 1.5rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .stButton > button:hover {{
        background: linear-gradient(45deg, {accent_color}, {primary_color});
        transform: translateY(-2px);
    }}
    
    .sidebar-header {{
        color: {primary_color};
        font-family: 'Orbitron', monospace;
        font-weight: 700;
        font-size: 1.3rem;
        text-align: center;
        margin-bottom: 1rem;
    }}
    
    .feature-list {{
        background: {surface_color};
        border-left: 3px solid {primary_color};
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
        font-family: 'Rajdhani', sans-serif;
        color: {text_color};
    }}
    
    .status-success {{
        color: {primary_color};
        background: {surface_color};
        padding: 0.5rem 1rem;
        border-radius: 6px;
        border-left: 4px solid {primary_color};
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
    }}
    
    .status-error {{
        color: {accent_color};
        background: {surface_color};
        padding: 0.5rem 1rem;
        border-radius: 6px;
        border-left: 4px solid {accent_color};
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
    }}
    
    .welcome-container {{
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, {surface_color}, transparent);
        border-radius: 20px;
        margin: 2rem 0;
    }}
    
    .welcome-title {{
        font-family: 'Orbitron', monospace;
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, {primary_color}, {secondary_color}, {accent_color});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }}
    
    .welcome-subtitle {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.3rem;
        color: {text_color};
        opacity: 0.8;
        margin-bottom: 2rem;
    }}
    
    .feature-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }}
    
    .feature-card {{
        background: {surface_color};
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid {border_color};
        text-align: left;
        transition: transform 0.3s ease;
    }}
    
    .feature-card:hover {{
        transform: translateY(-5px);
    }}
    
    .feature-card h3 {{
        color: {primary_color};
        font-family: 'Orbitron', monospace;
        margin-bottom: 1rem;
    }}
    
    {animation_css}
    </style>
    """

def welcome_page():
    """Displays a streamlit-native welcome page"""
    
    # Apply theme CSS
    st.markdown(get_theme_css(st.session_state.dark_mode, st.session_state.animations_enabled), unsafe_allow_html=True)
    
    # Welcome content
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-title">🚀 KRYPTONIC AI 🤖</div>
        <div class="welcome-subtitle">
            ⚡ Your Crypto Buddy That Actually Gets It ⚡<br>
            Navigate the digital financial frontier with intelligence, precision, and style.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the start button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 START CHATTING", key="start_chat", help="Begin your crypto journey!"):
            st.session_state.page = "chat"
            st.rerun()
    
    # Features grid
    st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 Smart AI Assistant</h3>
            <p>Get crypto explanations in simple terms, market insights, and personalized guidance from our AI that understands the Caribbean market.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>⚡ Real-Time Data</h3>
            <p>Live crypto prices, trending coins, market overviews, and up-to-the-minute data to keep you informed.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>🔒 Safe & Focused</h3>
            <p>Only crypto-focused conversations, honest risk discussions, and educational content designed for young investors.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def chat_page():
    """Displays the main chat interface"""
    
    # Apply theme CSS for the chat page
    st.markdown(get_theme_css(st.session_state.dark_mode, st.session_state.animations_enabled), unsafe_allow_html=True)
    
    # Custom header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="main-header">
            🚀 Kryptonic AI 🤖
        </div>
        <div class="sub-header">
            ⚡ Your Crypto Buddy That Actually Gets It ⚡<br>
            <span style="color: #00ff88;">🔮 Smart AI • Live Prices • Easy Explanations 🔮</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Add back to welcome button
    if st.button("← Back to Welcome", key="back_to_welcome"):
        st.session_state.page = "welcome"
        st.rerun()
    
    # Check if API key is configured
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        st.markdown("""
        <div class="status-error">
        ❌ <strong>OOPS! NEED TO SET UP API</strong><br>
        Need your Google AI key to get this working
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-list">
        <strong>🔑 Quick Setup:</strong><br>
        1. Go to → https://makersuite.google.com/app/apikey<br>
        2. Sign in with Google<br>
        3. Create a new API key<br>
        4. Put it in the code where it says "YOUR_GEMINI_API_KEY_HERE"<br>
        5. You're good to go! 🚀
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Sidebar with controls and quick actions
    with st.sidebar:
        st.markdown('<div class="sidebar-header">⚙ CONTROLS ⚙</div>', unsafe_allow_html=True)
        
        # Theme and animation controls
        col1, col2 = st.columns(2)
        with col1:
            dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="dark_toggle")
        with col2:
            animations = st.toggle("✨ Animations", value=st.session_state.animations_enabled, key="anim_toggle")
        
        # Update session state and rerun if changed
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            st.rerun()
        
        if animations != st.session_state.animations_enabled:
            st.session_state.animations_enabled = animations
            st.rerun()
        
        st.markdown("---")
        st.markdown('<div class="sidebar-header">⚡ QUICK STUFF ⚡</div>', unsafe_allow_html=True)
        
        # Initialize chatbot
        if 'chatbot' not in st.session_state:
            try:
                st.session_state.chatbot = CryptoChatbot(GEMINI_API_KEY)
                st.markdown("""
                <div class="status-success glow-text">
                ✅ CryptoMind AI: Ready!<br>
                🧠 AI brain loaded<br>
                🔗 Connected to crypto data
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class="status-error">
                ❌ <strong>UH OH!</strong><br>
                Something's not working: {str(e)}<br><br>
                <strong>Try This:</strong><br>
                • Check your internet<br>
                • Make sure the API key is right<br>
                • Refresh the page
                </div>
                """, unsafe_allow_html=True)
                return
        
        st.markdown("---")
        
        # Quick action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔥 TRENDING", key="trending_btn"):
                response = st.session_state.chatbot.handle_trending_query()
                if 'messages' not in st.session_state:
                    st.session_state.messages = []
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
            
            if st.button("💰 BTC PRICE", key="btc_btn"):
                response = st.session_state.chatbot.handle_price_query("bitcoin")
                if 'messages' not in st.session_state:
                    st.session_state.messages = []
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        with col2:
            if st.button("📊 TOP COINS", key="top_coins_btn"):
                response = st.session_state.chatbot.handle_market_query()
                if 'messages' not in st.session_state:
                    st.session_state.messages = []
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
            
            if st.button("🔄 CLEAR CHAT", key="clear_btn"):
                st.session_state.messages = []
                st.rerun()
        
        st.markdown("---")
        
        # Features section
        st.markdown("""
        <div class="feature-list">
        <div class="sidebar-header">🛡 WHAT I CAN DO</div>
        
        <strong>🤖 Smart Crypto Help</strong><br>
        • Explain crypto in simple terms<br>
        • Help you understand the market<br>
        • Answer all your crypto questions<br><br>
        
        <strong>⚡ Live Data</strong><br>
        • Real-time prices<br>
        • What's trending now<br>
        • Market overviews<br><br>
        
        <strong>🔒 Safe Space</strong><br>
        • Only talk about crypto<br>
        • No weird stuff<br>
        • Always honest about risks<br>
        </div>
        """, unsafe_allow_html=True)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages using Streamlit's built-in chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("🤑 Ask me anything about crypto..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Thinking about your question..."):
                response = st.session_state.chatbot.process_query(prompt)
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    # Initialize session state
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = True
    if 'animations_enabled' not in st.session_state
