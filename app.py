import streamlit as st
import requests
from datetime import datetime, date, timedelta
import pandas as pd
import google.generativeai as genai
import re
import time
import json
import os
from typing import List, Dict, Any
import random
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# --- Define Colors and Styles ---
BG_GRADIENT_COLOR_1 = "#2a1a4a"
BG_GRADIENT_COLOR_2 = "#1e1538"
BG_GRADIENT_COLOR_3 = "#161028"
BG_GRADIENT_COLOR_4 = "#0f0818"
BG_GRADIENT_COLOR_5 = "#000000"

TEXT_COLOR = "white"
BUTTON_BG_COLOR = "white"
BUTTON_TEXT_COLOR = "#1f2937"
HIGHLIGHT_GRADIENT_START = "#8b5cf6"
HIGHLIGHT_GRADIENT_MIDDLE = "#3b82f6"
HIGHLIGHT_GRADIENT_END = "#ec4899"
STARS_COLOR = "#fbbf24"

# Chatbot specific colors
PRIMARY_COLOR = "#00ff88"
SECONDARY_COLOR = "#00d4ff"
ACCENT_COLOR = "#ff0080"
SURFACE_COLOR = "rgba(255, 255, 255, 0.05)"
BORDER_COLOR = "rgba(0, 255, 136, 0.3)"

# API Configuration
GEMINI_API_KEY = "AIzaSyClYtKJRhSgTiua0OwwQQ9jU021mahtsSY"
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Kryptonic AI - Your Crypto Guide",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Expanded list of supported languages ---
LANGUAGE_OPTIONS = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Chinese (Simplified)': 'zh-Hans',
    'Arabic': 'ar',
    'Hindi': 'hi',
    'Bengali': 'bn',
    'Indonesian': 'id',
    'Dutch': 'nl',
    'Polish': 'pl',
    'Thai': 'th',
    'Turkish': 'tr',
    'Vietnamese': 'vi',
    'Romanian': 'ro',
    'Ukrainian': 'uk'
}

def translate_text(text: str, target_language_code: str) -> str:
    """
    Translates text to a target language using the Gemini API.
    
    Args:
        text (str): The text to be translated.
        target_language_code (str): The language code (e.g., 'en', 'es', 'fr').
        
    Returns:
        str: The translated text, or a fallback message if translation fails.
    """
    if target_language_code == 'en':
        return text

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        prompt = f"""
        Translate the following text into the language with the code '{target_language_code}':
        
        TEXT TO TRANSLATE:
        "{text}"
        
        Translated Text:
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return remove_html_tags(response.text)
    except Exception:
        return f"Translation failed. Original response: {text}"

def remove_emojis(text):
    """Remove Unicode emojis from text for text-to-speech."""
    if not text:
        return text
    
    # Unicode ranges for emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002600-\U000026FF"  # miscellaneous symbols
        "\U00002700-\U000027BF"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "\U00002B50-\U00002B55"  # stars
        "]+", 
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def remove_html_tags(text):
    """Remove all HTML tags and clean up the response."""
    if not text:
        return text
    
    # Remove HTML tags
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    
    # Remove common HTML entities
    html_entities = {
        '&nbsp;': ' ',
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&hellip;': '...',
    }
    
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)
    
    # Remove any remaining markdown that could create HTML
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove **bold**
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove *italic*
    text = re.sub(r'`(.*?)`', r'\1', text)        # Remove `code`
    
    # Clean up extra whitespace
    text = ' '.join(text.split())
    
    return text

class CryptoChatbot:
    """A chatbot class that fetches real-time crypto data and answers user queries with voice features."""
    def __init__(self, gemini_api_key):
        # Configure Gemini API
        genai.configure(api_key=gemini_api_key)
        
        # Initialize the generative model
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception:
            try:
                self.model = genai.GenerativeModel('gemini-1.0-pro')
            except Exception:
                self.model = genai.GenerativeModel('models/gemini-pro')
        
        self.supported_coins = [
            'bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana',
            'polkadot', 'dogecoin', 'avalanche-2', 'chainlink', 'polygon',
            'ripple', 'litecoin', 'stellar', 'monero', 'tron'
        ]

        # Load CSV data for Q&A functionality
        self.csv_data = self.load_csv_data()
        
        # Voice processing patterns for command recognition
        self.voice_commands = {
            'price': ['price', 'cost', 'value', 'worth', 'trading at'],
            'chart': ['chart', 'graph', 'show', 'display', 'visualize'],
            'compare': ['compare', 'comparison', 'versus', 'vs', 'against'],
            'trending': ['trending', 'popular', 'hot', 'top'],
            'help': ['help', 'assistance', 'guide', 'how to'],
            'explain': ['explain', 'what is', 'tell me about', 'describe']
        }
    
    def load_csv_data(self):
        """Load the cryptocurrency Q&A data from CSV file."""
        try:
            csv_path = os.path.join("c:\\Users\\Verona\\Desktop\\eccu camp", "csv_files", "Crypto Currency.csv")
            df = pd.read_csv(csv_path)
            # Clean the data - remove empty rows and ensure we have the required columns
            df = df.dropna(subset=['question', 'answer'])
            df = df[df['question'].str.strip() != '']
            df = df[df['answer'].str.strip() != '']
            return df
        except Exception as e:
            print(f"Warning: Could not load CSV data: {e}")
            return pd.DataFrame(columns=['question', 'answer', 'topic', 'price'])
    
    def search_csv_knowledge(self, user_question):
        """Search the CSV data for relevant Q&A pairs."""
        if self.csv_data.empty:
            return None
            
        user_question_lower = user_question.lower()
        best_matches = []
        
        for index, row in self.csv_data.iterrows():
            question = str(row['question']).lower()
            answer = str(row['answer'])
            
            # Simple keyword matching - can be enhanced with more sophisticated methods
            keywords_in_question = user_question_lower.split()
            matches = sum(1 for keyword in keywords_in_question if keyword in question)
            
            if matches > 0:
                # Calculate match score
                score = matches / len(keywords_in_question)
                best_matches.append({
                    'question': row['question'],
                    'answer': answer,
                    'score': score,
                    'topic': row.get('topic', 'Cryptocurrency')
                })
        
        # Sort by score and return the best match if score is good enough
        if best_matches:
            best_matches.sort(key=lambda x: x['score'], reverse=True)
            best_match = best_matches[0]
            
            # Only return if the match score is reasonable (at least 20% of words match)
            if best_match['score'] >= 0.2:
                return best_match
                
        return None
    
    def get_crypto_price(self, coin_id, use_cache=True):
        """Get current price and basic info for a cryptocurrency with caching."""
        # Initialize cache in session state
        if 'price_cache' not in st.session_state:
            st.session_state.price_cache = {}
        if 'last_api_call' not in st.session_state:
            st.session_state.last_api_call = 0
        
        # Check cache first (cache for 30 seconds)
        cache_key = coin_id
        current_time = time.time()
        
        if use_cache and cache_key in st.session_state.price_cache:
            cached_data, cache_time = st.session_state.price_cache[cache_key]
            if current_time - cache_time < 30:  # Use cached data for 30 seconds
                return cached_data
        
        # Rate limiting: check if we can make a call
        time_since_last_call = current_time - st.session_state.last_api_call
        if time_since_last_call < 2:
            st.toast(f"⏳ API rate limit active. Please wait {2 - time_since_last_call:.1f}s.", icon="⏳")
            # Return cached data if available, otherwise nothing
            if cache_key in st.session_state.price_cache:
                cached_data, _ = st.session_state.price_cache[cache_key]
                return cached_data
            return None
        
        try:
            url = f"{COINGECKO_BASE_URL}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true'
            }
            
            with st.spinner(f"⏳ Fetching {coin_id} data..."):
                response = requests.get(url, params=params, timeout=15)
                st.session_state.last_api_call = time.time()
                
                if response.status_code == 429:
                    st.error("🚫 **Rate Limit Exceeded!** Please wait a moment before trying again.")
                    # Return cached data if available
                    if cache_key in st.session_state.price_cache:
                        st.warning("📋 Using cached data instead...")
                        cached_data, _ = st.session_state.price_cache[cache_key]
                        return cached_data
                    return None
                
                response.raise_for_status()
                data = response.json()
                
                # Cache the successful response
                st.session_state.price_cache[cache_key] = (data, current_time)
                
                return data
                
        except requests.exceptions.RequestException as e:
            if "429" in str(e):
                st.error("🚫 **Rate Limit Exceeded!** CoinGecko free API allows ~10-50 calls per minute. Please wait before trying again.")
                st.info("💡 **Tip:** Try refreshing in 1-2 minutes or use the cached data.")
            else:
                st.error(f"❌ Error fetching data for {coin_id}: {e}")
            
            # Return cached data if available
            if cache_key in st.session_state.price_cache:
                st.warning("📋 Using cached data instead...")
                cached_data, _ = st.session_state.price_cache[cache_key]
                return cached_data
            return None
        except Exception as e:
            st.error(f"❌ Unexpected error fetching data for {coin_id}: {e}")
            return None
    
    def get_trending_coins(self):
        """Get trending cryptocurrencies."""
        try:
            url = f"{COINGECKO_BASE_URL}/search/trending"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Error fetching trending coins: {e}")
            return None
    
    def get_market_overview_df(self):
        """Get top cryptocurrencies by market cap and return a DataFrame."""
        try:
            url = f"{COINGECKO_BASE_URL}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': 10,
                'page': 1,
                'sparkline': 'false'
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            market_data = response.json()
            
            df = pd.DataFrame(market_data)
            df_display = df[['name', 'symbol', 'current_price', 'price_change_percentage_24h', 'market_cap']].copy()
            df_display.columns = ['Coin', 'Symbol', 'Price', '24h Change', 'Market Cap']
            df_display['Price'] = df_display['Price'].apply(lambda x: f"${x:,.2f}")
            df_display['24h Change'] = df_display['24h Change'].apply(lambda x: f"{x:+.2f}%")
            df_display['Market Cap'] = df_display['Market Cap'].apply(lambda x: f"${x:,.0f}")
            
            return df_display
        except Exception as e:
            st.error(f"Error fetching market overview: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, coin_id, days=30):
        """Get historical price data for interactive charts with caching."""
        # Initialize cache
        if 'historical_cache' not in st.session_state:
            st.session_state.historical_cache = {}
        
        # Check cache first (cache for 5 minutes for historical data)
        cache_key = f"{coin_id}_{days}"
        current_time = time.time()
        
        if cache_key in st.session_state.historical_cache:
            cached_data, cache_time = st.session_state.historical_cache[cache_key]
            if current_time - cache_time < 300:  # Use cached data for 5 minutes
                return cached_data
        
        # Rate limiting check
        time_since_last_call = current_time - st.session_state.get('last_api_call', 0)
        if time_since_last_call < 3:  # Wait 3 seconds for historical data
            st.toast(f"⏳ Chart API rate limit active. Please wait {3 - time_since_last_call:.1f}s.", icon="⏳")
            # Return cached data if available, otherwise an empty dataframe
            if cache_key in st.session_state.historical_cache:
                cached_data, _ = st.session_state.historical_cache[cache_key]
                return cached_data
            return pd.DataFrame()

        try:
            url = f"{COINGECKO_BASE_URL}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly' if days <= 7 else 'daily'
            }
            
            with st.spinner(f"📊 Loading {days}-day chart data for {coin_id.title()}..."):
                response = requests.get(url, params=params, timeout=20)
                st.session_state.last_api_call = time.time()
                
                if response.status_code == 429:
                    st.error("🚫 **Rate Limit Exceeded!** Please wait before loading charts.")
                    # Return cached data if available
                    if cache_key in st.session_state.historical_cache:
                        st.warning("📋 Using cached chart data...")
                        cached_data, _ = st.session_state.historical_cache[cache_key]
                        return cached_data
                    return pd.DataFrame()
                
                response.raise_for_status()
                data = response.json()
                
                # Convert to DataFrame
                prices = data['prices']
                volumes = data['total_volumes']
                market_caps = data['market_caps']
                
                df = pd.DataFrame({
                    'timestamp': [datetime.fromtimestamp(price[0]/1000) for price in prices],
                    'price': [price[1] for price in prices],
                    'volume': [vol[1] for vol in volumes],
                    'market_cap': [mc[1] for mc in market_caps]
                })
                
                # Cache the result
                st.session_state.historical_cache[cache_key] = (df, current_time)
                
                return df
                
        except requests.exceptions.RequestException as e:
            if "429" in str(e):
                st.error("🚫 **Chart Rate Limited!** Please wait 1-2 minutes before loading charts.")
                st.info("💡 CoinGecko limits chart requests. Try again shortly.")
            else:
                st.error(f"❌ Error fetching historical data for {coin_id}: {e}")
            
            # Return cached data if available
            if cache_key in st.session_state.historical_cache:
                st.warning("📋 Using cached chart data...")
                cached_data, _ = st.session_state.historical_cache[cache_key]
                return cached_data
            return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Unexpected error loading chart data: {e}")
            return pd.DataFrame()
    
    def create_mock_chart_data(self, coin_id, coin_name, days=7):
        """Generate mock chart data when API fails."""
        try:
            # Get current price first for realistic data
            price_data = self.get_crypto_price(coin_id, use_cache=True)
            if price_data and coin_id in price_data:
                current_price = price_data[coin_id]['usd']
            else:
                # Fallback prices for common coins
                fallback_prices = {
                    'bitcoin': 45000,
                    'ethereum': 3000,
                    'binancecoin': 300,
                    'cardano': 0.5,
                    'solana': 100,
                    'polkadot': 25,
                    'dogecoin': 0.08,
                    'chainlink': 15
                }
                current_price = fallback_prices.get(coin_id, 100)
            
            # Generate realistic price movements
            timestamps = []
            prices = []
            volumes = []
            
            import random
            from datetime import datetime, timedelta
            
            base_price = current_price
            for i in range(days * 24):  # Hourly data
                timestamp = datetime.now() - timedelta(hours=days*24-i)
                timestamps.append(timestamp)
                
                # Add realistic price variation (±2% per hour)
                variation = random.uniform(-0.02, 0.02)
                base_price = base_price * (1 + variation)
                prices.append(max(base_price, 0.01))  # Ensure positive prices
                
                # Generate volume data
                volume = random.uniform(1000000, 10000000)
                volumes.append(volume)
            
            df = pd.DataFrame({
                'timestamp': timestamps,
                'price': prices,
                'volume': volumes
            })
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error generating mock data: {e}")
            return pd.DataFrame()
    
    def create_simple_price_chart(self, coin_id, coin_name, df):
        """Create a simple, reliable price chart."""
        try:
            fig = go.Figure()
            
            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['price'],
                    mode='lines+markers',
                    name=f'{coin_name.title()} Price',
                    line=dict(color=PRIMARY_COLOR, width=3),
                    marker=dict(size=4, color=PRIMARY_COLOR),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Time: %{x}<br>' +
                                 'Price: $%{y:,.4f}<br>' +
                                 '<extra></extra>'
                )
            )
            
            # Calculate stats
            current_price = df['price'].iloc[-1]
            start_price = df['price'].iloc[0]
            price_change = current_price - start_price
            price_change_pct = (price_change / start_price) * 100 if start_price != 0 else 0
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=f"{coin_name.title()} - ${current_price:,.4f} ({price_change:+.4f}, {price_change_pct:+.2f}%)",
                    font=dict(color=TEXT_COLOR, size=18),
                    x=0.5
                ),
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)',
                font=dict(color=TEXT_COLOR),
                height=500,
                showlegend=True,
                xaxis=dict(
                    title="Time",
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                yaxis=dict(
                    title="Price (USD)",
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    tickformat='$,.4f'
                ),
                margin=dict(l=50, r=50, t=60, b=50)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"❌ Error creating simple chart: {e}")
            return None

    def create_interactive_chart(self, coin_id, coin_name, days=7):
        """Create an interactive Plotly chart with robust fallbacks."""
        try:
            # Try to get real data first
            df = self.get_historical_data(coin_id, days)
            data_source = "live"
            
            # If real data fails, try mock data
            if df.empty:
                st.warning(f"📊 Live data unavailable for {coin_name.title()}. Showing simulated data based on current price.")
                st.info("💡 This may be due to API rate limiting. Try again in 1-2 minutes for live data.")
                df = self.create_mock_chart_data(coin_id, coin_name, days)
                data_source = "simulated"
                
                if df.empty:
                    st.error("❌ Unable to create chart data. Please try a different cryptocurrency.")
                    return None
            
            # Validate we have essential data
            if len(df) < 2:
                st.warning(f"⚠️ Insufficient data points ({len(df)} points)")
                return None
            
            # Create simple chart that's more reliable
            fig = self.create_simple_price_chart(coin_id, coin_name, df)
            
            if fig:
                # Add data source indicator
                if data_source == "simulated":
                    fig.add_annotation(
                        text="⚠️ Simulated Data",
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        showarrow=False,
                        font=dict(size=12, color="orange"),
                        bgcolor="rgba(255,165,0,0.2)",
                        bordercolor="orange"
                    )
                else:
                    fig.add_annotation(
                        text="✅ Live Data",
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        showarrow=False,
                        font=dict(size=12, color=PRIMARY_COLOR),
                        bgcolor="rgba(0,255,136,0.2)",
                        bordercolor=PRIMARY_COLOR
                    )
                
                return fig
            else:
                return None
            
        except Exception as e:
            st.error(f"❌ Error creating chart for {coin_name.title()}: {str(e)}")
            
            # Last resort fallback - show current price info
            st.info("🔄 **Fallback:** Showing current price information instead:")
            self.show_chart_fallback(coin_name, coin_id)
            return None
    
    def create_comparison_chart(self, coin_ids, coin_names, days=7):
        """Create a comparison chart for multiple cryptocurrencies with robust fallbacks."""
        try:
            fig = go.Figure()
            colors = [PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, "#fbbf24", "#f59e0b"]
            successful_coins = []
            failed_coins = []
            has_live_data = False
            has_simulated_data = False
            
            for i, (coin_id, coin_name) in enumerate(zip(coin_ids, coin_names)):
                try:
                    # Try real data first
                    df = self.get_historical_data(coin_id, days)
                    data_source = "live"
                    
                    # If real data fails, try simulated data
                    if df.empty:
                        df = self.create_mock_chart_data(coin_id, coin_name, days)
                        data_source = "simulated"
                        has_simulated_data = True
                    else:
                        has_live_data = True
                    
                    if not df.empty and 'price' in df.columns and len(df) >= 2:
                        # Safely normalize prices to percentage change from start
                        start_price = float(df['price'].iloc[0])
                        if start_price != 0:
                            df['normalized'] = ((df['price'] - start_price) / start_price) * 100
                            
                            # Adjust line style based on data source
                            line_style = dict(color=colors[i % len(colors)], width=2)
                            if data_source == "simulated":
                                line_style['dash'] = 'dash'  # Dashed line for simulated data
                            
                            coin_display_name = f"{coin_name.title()}"
                            if data_source == "simulated":
                                coin_display_name += " (sim)"
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=df['timestamp'],
                                    y=df['normalized'],
                                    mode='lines',
                                    name=coin_display_name,
                                    line=line_style,
                                    hovertemplate=f'<b>{coin_display_name}</b><br>' +
                                                 'Time: %{x}<br>' +
                                                 'Change: %{y:.2f}%<br>' +
                                                 '<extra></extra>'
                                )
                            )
                            successful_coins.append(coin_display_name)
                        else:
                            failed_coins.append(f"{coin_name.title()} (zero price)")
                    else:
                        failed_coins.append(f"{coin_name.title()} (no data)")
                        
                except Exception as e:
                    failed_coins.append(f"{coin_name.title()} (error: {str(e)[:30]})")
                    continue
            
            # Show status messages
            if successful_coins:
                st.success(f"✅ Successfully loaded: {', '.join(successful_coins)}")
                
                # Show data source info
                if has_simulated_data and has_live_data:
                    st.info("📊 Mixed data: Live data (solid lines) and simulated data (dashed lines)")
                elif has_simulated_data and not has_live_data:
                    st.warning("⚠️ All data is simulated due to API limitations. Try again in 1-2 minutes for live data.")
                elif has_live_data and not has_simulated_data:
                    st.success("✅ All data is live from the market")
            
            if failed_coins:
                st.warning(f"⚠️ Failed to load: {', '.join(failed_coins)}")
                st.info("💡 This could be due to API rate limits or data unavailability")
            
            # Only return chart if we have at least one successful trace
            if len(fig.data) == 0:
                st.error("❌ No data available for any of the selected cryptocurrencies")
                return None
            
            fig.update_layout(
                title=dict(
                    text=f"Price Comparison - Last {days} Days",
                    font=dict(color=TEXT_COLOR, size=20),
                    x=0.5
                ),
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=TEXT_COLOR),
                height=500,
                showlegend=True,
                xaxis_title="Time",
                yaxis_title="Price Change (%)",
                yaxis=dict(tickformat='+.2f%'),
                margin=dict(l=0, r=0, t=50, b=0)
            )
            
            fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            
            return fig
            
        except Exception as e:
            st.error(f"❌ Error creating comparison chart: {str(e)}")
            st.info("💡 **Troubleshooting Tips:**")
            st.info("• Try with fewer cryptocurrencies")
            st.info("• Wait 1-2 minutes if you've been making many requests")
            st.info("• Check your internet connection")
            return None
            
    def get_current_market_data(self, num_top_coins=5, num_trending_coins=3):
        """Get current market data to provide context to AI."""
        market_data = self.get_market_overview_df()
        trending_data = self.get_trending_coins()
        
        context = "Current Crypto Market Data:\n"
        
        if not market_data.empty:
            context += f"Top {num_top_coins} Cryptocurrencies by Market Cap:\n"
            for i, row in market_data.head(num_top_coins).iterrows():
                context += f"{i+1}. {row['Coin']} ({row['Symbol']}): {row['Price']} ({row['24h Change']})\n"
        
        if trending_data and 'coins' in trending_data:
            context += f"\nTrending Coins (Top {num_trending_coins}):\n"
            for i, coin in enumerate(trending_data['coins'][:num_trending_coins], 1):
                context += f"{i}. {coin['item']['name']} ({coin['item']['symbol']})\n"
        
        return context
    
    def build_conversation_context(self, messages):
        """Build conversation context from message history."""
        if not messages:
            return ""
        
        context = "Previous conversation context:\n"
        # Only include the last 10 messages to avoid token limits
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        
        for msg in recent_messages:
            if msg["role"] == "user":
                context += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant" and msg.get("type") != "dataframe":
                # Truncate long responses for context
                content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
                context += f"Kryptonic: {content}\n"
        
        return context
    
    def ask_ai(self, user_question, conversation_history):
        """
        Ask Gemini AI about cryptocurrency topics with conversation memory.
        
        The model now generates a response in English, which is then translated.
        This removes ambiguity from the prompt and ensures consistent translation.
        """
        try:
            # First, search CSV knowledge base for relevant answers
            csv_match = self.search_csv_knowledge(user_question)
            
            market_context = self.get_current_market_data()
            conversation_context = self.build_conversation_context(conversation_history)
            
            # Check if this is the first interaction
            is_first_interaction = len([msg for msg in conversation_history if msg["role"] == "user"]) == 0
            
            base_prompt = f"""You are "Kryptonic," a young, super intelligent cryptocurrency expert with a Gen Z persona. Your goal is to be an informative, friendly, encouraging, and fun guide for beginners learning about crypto. You can use Gen Z slang but still sound professional and knowledgeable with a casual, friendly, conversational tone. Your responses should be easy to understand and avoid overwhelming jargon.

IMPORTANT CONVERSATION RULES:
1. {"This is the user's FIRST question - introduce yourself as Kryptonic and welcome them warmly." if is_first_interaction else "You are CONTINUING an ongoing conversation - DO NOT introduce yourself again. The user already knows who you are."}
2. Be natural and conversational - refer to previous topics if relevant
3. Don't repeat information you've already shared unless specifically asked

PERSONALITY:
- Be friendly, enthusiastic, and relatable
- Use simple, everyday language (no fancy financial jargon)
- Be like a knowledgeable friend explaining crypto
- Use emojis but don't overdo it
- Keep it real and honest about risks
- Make complex topics easy to understand

SPECIAL HANDLING:
1. If someone asks "who are you", "what are you", "tell me about yourself" - explain you're Kryptonic AI, a crypto expert that helps with crypto questions, prices, and education
2. If someone asks "what can you do", "how can you help" - list your capabilities: crypto explanations, live prices, market data, trading help, etc.
3. For basic conversational stuff like "thanks", "goodbye" - respond appropriately but guide back to crypto
4. If it's completely unrelated to crypto (weather, sports, etc.) - politely redirect: "Hey! I'm all about crypto stuff. What crypto question can I help with?"

RULES:
1. Focus primarily on crypto-related topics If asked respond with: "I'm so sorry, but I do not have the capabilities to answer this question. I can however assist with any and everything cryptocurrency related!
2. Handle basic greetings and questions about yourself naturally
3. Explain things simply but accurately and professionally 
4. Use current market data when it helps
5. Keep responses under 150 words
6. Be encouraging but realistic about crypto investing
7. Always mention that crypto is risky when giving advice
8. Answer in point form where necessary
9. *CRITICAL: NEVER use HTML tags, XML tags, or any markup in your response. No <div>, </div>, <span>, <p>, <br>, or ANY tags starting with < or ending with >. Use only plain text with emojis.*
10. Never be rude or disrespectful
11. Never use inappropriate language
12. Do not encourage or give advice on illegal actions
13. Never give personal opinions to the user
14. Always provide factual information

TONE EXAMPLES:
- Instead of "utilize" say "use"
- Instead of "substantial" say "big" or "huge"
- Instead of "fluctuations" say "price changes"
- Instead of "portfolio diversification" say "spreading your money around"

{conversation_context}

Current Market Context:
{market_context}

{f'''IMPORTANT: I found relevant information in my knowledge base that directly addresses this question:
Question: {csv_match["question"]}
Answer: {csv_match["answer"]}

Use this information as the foundation for your response, but adapt it to match your personality and style. You can expand on it, simplify it, or add current market context if relevant.''' if csv_match else ''}

User Question: {user_question}

Give a helpful, friendly and professional response:"""

            response = self.model.generate_content(base_prompt)
            
            # Multiple cleaning steps to ensure no HTML appears
            clean_response = response.text
            
            # Step 1: Remove HTML tags
            clean_response = remove_html_tags(clean_response)
            
            # Step 2: Additional safety cleaning for common HTML patterns
            html_patterns = [
                r'<[^>]*>',      # Any HTML tags
                r'</[^>]*>',     # Any closing HTML tags
                r'&[a-zA-Z]+;',  # HTML entities
                r'&#\d+;',       # Numeric HTML entities
            ]
            
            for pattern in html_patterns:
                clean_response = re.sub(pattern, '', clean_response)
            
            # Step 3: Clean up spacing after removal
            clean_response = ' '.join(clean_response.split())
            
            return clean_response
            
        except Exception as e:
            return f"Oops! Something went wrong on my end 😅 Try asking again in a second: {str(e)}"
    
    def handle_price_query(self, coin_id):
        """Handle price-related queries - same as app.py style"""
        data = self.get_crypto_price(coin_id)
        if data and coin_id in data:
            coin_data = data[coin_id]
            price = coin_data.get('usd', 0)
            change_24h = coin_data.get('usd_24h_change', 0)
            market_cap = coin_data.get('usd_market_cap', 0)
            volume_24h = coin_data.get('usd_24h_vol', 0)
            
            response_str = f"""{coin_id.replace('-', ' ').title()} Right Now 💰

💵 Price: ${price:,.2f}
{"📈" if change_24h > 0 else "📉"} 24h: {change_24h:+.2f}%
📊 Market Cap: ${market_cap:,.0f}
💹 Daily Trading: ${volume_24h:,.0f}

Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC

Remember: Crypto prices change super fast! ⚡"""
            return response_str
        else:
            return f"Hmm, couldn't grab the price for {coin_id} right now 🤔 Maybe try again in a bit?"
    
    def handle_trending_query(self):
        """Handle trending coins query - same as app.py style"""
        trending_data = self.get_trending_coins()
        if trending_data and 'coins' in trending_data:
            trending_coins = trending_data['coins'][:5]
            response = "🔥 What's Hot Right Now:\n\n"
            for i, coin in enumerate(trending_coins, 1):
                response += f"{i}. {coin['item']['name']} ({coin['item']['symbol'].upper()})\n"
                response += f"    Market Rank: #{coin['item']['market_cap_rank']}\n\n"
            response += "These are the coins everyone's talking about today! 🚀"
            return response
        else:
            return "Can't get the trending list right now 😕 Try again in a moment!"
    
    def process_query(self, user_input, language_code, conversation_history):
        """Process user query and return appropriate response."""
        user_input_lower = user_input.lower()
        
        # 1. Check for price queries
        if "price" in user_input_lower and any(coin.replace('-', '').replace('2', '') in user_input_lower.replace(' ', '') for coin in self.supported_coins):
            for coin in self.supported_coins:
                if coin.replace('-', '').replace('2', '') in user_input_lower.replace(' ', ''):
                    response = self.handle_price_query(coin)
                    return translate_text(response, language_code)
        
        # 2. Check for trending queries
        elif "trending" in user_input_lower or "popular" in user_input_lower:
            response = self.handle_trending_query()
            return translate_text(response, language_code)
        
        # 3. Check for market overview queries
        elif "market" in user_input_lower and ("overview" in user_input_lower or "top" in user_input_lower):
            return "market_overview_requested"  # Special flag for handling in main function
        
        # 4. Fallback to the generative model for all other questions
        else:
            # The AI model generates a response in English.
            response = self.ask_ai(user_input, conversation_history)
            # This response is then translated into the selected language.
            return translate_text(response, language_code)
    
    def show_api_diagnostic(self):
        """Display diagnostic information about API status and usage."""
        st.info("🔍 **API Diagnostic Information**")
        
        # Check cache status
        cache_info = []
        if hasattr(st.session_state, 'price_cache'):
            cache_info.append(f"Price cache: {len(st.session_state.price_cache)} items")
        if hasattr(st.session_state, 'historical_cache'):
            cache_info.append(f"Chart cache: {len(st.session_state.historical_cache)} items")
        
        if cache_info:
            st.write("📋 **Cache Status:**", " | ".join(cache_info))
        
        # Check last API call time
        if hasattr(st.session_state, 'last_api_call'):
            time_since_last_call = time.time() - st.session_state.last_api_call
            st.write(f"⏰ **Last API call:** {time_since_last_call:.1f} seconds ago")
            
            if time_since_last_call < 2:
                st.warning("⚠️ Rate limit may be active. Wait before making new requests.")
            else:
                st.success("✅ Ready to make new API requests")
        else:
            st.write("⏰ **Last API call:** No recent calls")
        
        # Test basic connectivity
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧪 Test Price API"):
                test_price = self.get_crypto_price('bitcoin', use_cache=False)
                if test_price:
                    st.success("✅ Price API working")
                    st.write(f"Bitcoin: ${test_price['bitcoin']['usd']:,.2f}")
                else:
                    st.error("❌ Price API failed")
        
        with col2:
            if st.button("📊 Test Chart API"):
                test_data = self.get_historical_data('bitcoin', days=1)
                if not test_data.empty:
                    st.success("✅ Chart API working")
                    st.write(f"Retrieved {len(test_data)} data points")
                else:
                    st.error("❌ Chart API failed - will use simulated data")
        
        # Quick chart test
        if st.button("🧪 Test New Chart System", use_container_width=True):
            st.write("Testing improved chart system...")
            
            test_chart = self.create_interactive_chart('bitcoin', 'bitcoin', days=1)
            if test_chart:
                st.plotly_chart(test_chart, use_container_width=True)
                st.success("✅ New chart system working perfectly!")
            else:
                st.info("ℹ️ Chart system using fallback mode - this is normal when API is rate limited")
    
    def create_ascii_chart(self, coin_id, coin_name):
        """Create a simple ASCII-style price chart as ultimate fallback."""
        try:
            # Get current price
            price_data = self.get_crypto_price(coin_id)
            if not price_data or coin_id not in price_data:
                return None
            
            current_price = price_data[coin_id]['usd']
            change_24h = price_data[coin_id].get('usd_24h_change', 0)
            
            # Create simple visual representation
            trend_char = "📈" if change_24h > 0 else "📉" if change_24h < 0 else "➡️"
            bars = "█" * min(int(abs(change_24h)), 20)  # Max 20 bars
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {BG_GRADIENT_COLOR_1}, {BG_GRADIENT_COLOR_2}); 
                        padding: 1.5rem; border-radius: 10px; border: 1px solid {BORDER_COLOR};
                        font-family: monospace;">
                <h3 style="color: {PRIMARY_COLOR}; margin-bottom: 1rem;">{coin_name.title()} Price Overview</h3>
                <div style="color: {TEXT_COLOR}; font-size: 1.2rem; line-height: 1.8;">
                    💰 <strong>Current Price:</strong> ${current_price:,.4f}<br>
                    {trend_char} <strong>24h Change:</strong> {change_24h:+.2f}%<br>
                    📊 <strong>Visual Trend:</strong> {bars if bars else "━"}<br>
                    ⏰ <strong>Updated:</strong> Just now
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error creating fallback display: {e}")
            return None

    def show_chart_fallback(self, coin_name, coin_id):
        """Show alternative information when chart fails to load."""
        st.warning(f"📊 Chart temporarily unavailable for {coin_name.title()}")
        st.info("💡 **Alternative:** Here's the latest price information instead:")
        
        # Try ASCII chart first
        if self.create_ascii_chart(coin_id, coin_name):
            return
        
        # Try to get basic price data
        try:
            price_data = self.get_crypto_price(coin_id)
            if price_data and coin_id in price_data:
                coin_data = price_data[coin_id]
                current_price = coin_data['usd']
                change_24h = coin_data.get('usd_24h_change', 0)
                market_cap = coin_data.get('usd_market_cap', 0)
                volume_24h = coin_data.get('usd_24h_vol', 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("💰 Current Price", f"${current_price:,.2f}")
                    st.metric("💎 Market Cap", f"${market_cap:,.0f}")
                with col2:
                    st.metric("📈 24h Change", f"{change_24h:+.2f}%", delta=f"{change_24h:+.2f}%")
                    st.metric("📊 24h Volume", f"${volume_24h:,.0f}")
            else:
                st.error("❌ Unable to fetch any data for this cryptocurrency")
        except Exception as e:
            st.error(f"❌ Error fetching price data: {str(e)}")
        
        st.info("🔄 **Try these solutions:**")
        st.info("• Click 'REFRESH DATA' button above")
        st.info("• Wait 1-2 minutes if you've been making many requests")
        st.info("• Check the troubleshooting section in the sidebar")
        st.info("• Try a different cryptocurrency")
    
    def process_voice_command(self, voice_text):
        """Process voice input and identify command type with smart suggestions."""
        if not voice_text:
            return None, None, None
            
        voice_text_lower = voice_text.lower()
        
        # Extract cryptocurrency names
        detected_coins = []
        coin_aliases = {
            'btc': 'bitcoin', 'bitcoin': 'bitcoin',
            'eth': 'ethereum', 'ethereum': 'ethereum', 
            'bnb': 'binancecoin', 'binance coin': 'binancecoin',
            'ada': 'cardano', 'cardano': 'cardano',
            'sol': 'solana', 'solana': 'solana',
            'dot': 'polkadot', 'polkadot': 'polkadot',
            'doge': 'dogecoin', 'dogecoin': 'dogecoin',
            'link': 'chainlink', 'chainlink': 'chainlink'
        }
        
        for alias, coin_id in coin_aliases.items():
            if alias in voice_text_lower:
                detected_coins.append((alias.title(), coin_id))
        
        # Identify command type
        command_type = None
        for cmd, patterns in self.voice_commands.items():
            if any(pattern in voice_text_lower for pattern in patterns):
                command_type = cmd
                break
        
        # Generate smart response based on detected pattern
        suggestion = self.generate_voice_suggestion(command_type, detected_coins, voice_text)
        
        return command_type, detected_coins, suggestion
    
    def generate_voice_suggestion(self, command_type, detected_coins, original_text):
        """Generate smart suggestions for voice commands."""
        if not command_type and not detected_coins:
            return "💡 Try saying: 'What's the Bitcoin price?' or 'Show me Ethereum chart'"
        
        if command_type == 'price' and detected_coins:
            coin_name = detected_coins[0][0]
            return f"🎯 Getting {coin_name} price information..."
        
        if command_type == 'chart' and detected_coins:
            coin_name = detected_coins[0][0]
            return f"📊 Opening {coin_name} chart..."
        
        if command_type == 'trending':
            return "🔥 Loading trending cryptocurrencies..."
        
        if command_type == 'compare' and len(detected_coins) >= 2:
            coins = " vs ".join([coin[0] for coin in detected_coins[:2]])
            return f"⚖️ Comparing {coins}..."
        
        return "🎤 Processing your voice command..."
    
    def create_voice_input_component(self):
        """Create a voice input component using HTML5 Web Speech API."""
        voice_html = f"""
        <div style="background: linear-gradient(135deg, {BG_GRADIENT_COLOR_1}, {BG_GRADIENT_COLOR_2}); 
                    padding: 1.5rem; border-radius: 15px; border: 2px solid {PRIMARY_COLOR};
                    text-align: center; margin: 1rem 0;">
            <h3 style="color: {PRIMARY_COLOR}; margin-bottom: 1rem;">🎤 Voice Input</h3>
            
            <button id="voice-btn" onclick="toggleVoiceRecognition()" 
                    style="background: linear-gradient(45deg, {PRIMARY_COLOR}, {SECONDARY_COLOR}); 
                           border: none; padding: 1rem 2rem; border-radius: 10px; color: white; 
                           font-size: 1.1rem; cursor: pointer; margin: 0.5rem;">
                🎤 Start Voice Input
            </button>
            
            <div id="voice-status" style="color: {TEXT_COLOR}; margin-top: 1rem; font-size: 1rem;">
                Click the button and speak your crypto question
            </div>
            
            <div id="voice-result" style="background: rgba(0,255,136,0.1); padding: 1rem; 
                                         border-radius: 10px; margin-top: 1rem; color: {TEXT_COLOR}; 
                                         font-family: monospace; min-height: 2rem; display: none;">
            </div>
        </div>

        <script>
        let recognition;
        let isListening = false;

        function toggleVoiceRecognition() {{
            const btn = document.getElementById('voice-btn');
            const status = document.getElementById('voice-status');
            const result = document.getElementById('voice-result');
            
            if (!isListening) {{
                startVoiceRecognition(btn, status, result);
            }} else {{
                stopVoiceRecognition(btn, status);
            }}
        }}

        function startVoiceRecognition(btn, status, result) {{
            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {{
                recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
                recognition.continuous = false;
                recognition.interimResults = false;
                recognition.lang = 'en-US';

                recognition.onstart = function() {{
                    isListening = true;
                    btn.innerHTML = '🔴 Listening...';
                    btn.style.background = 'linear-gradient(45deg, #ff4444, #ff6666)';
                    status.innerHTML = '🎧 Listening... Speak now!';
                    result.style.display = 'none';
                }};

                recognition.onresult = function(event) {{
                    const transcript = event.results[0][0].transcript;
                    const confidence = event.results[0][0].confidence;
                    
                    let confidenceColor = confidence > 0.8 ? '{PRIMARY_COLOR}' : confidence > 0.6 ? '#fbbf24' : '#ff6b6b';
                    
                    result.innerHTML = `
                        <strong>You said:</strong> "${{transcript}}"<br>
                        <small style="color: ${{confidenceColor}};">Confidence: ${{Math.round(confidence * 100)}}%</small>
                    `;
                    result.style.display = 'block';
                    
                    // Send result to Streamlit with slight delay to ensure visibility
                    setTimeout(() => {{
                        const textArea = document.querySelector('[data-testid="stTextArea"] textarea');
                        if (textArea) {{
                            textArea.value = transcript;
                            textArea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            
                            // Auto-submit if confidence is high
                            if (confidence > 0.85) {{
                                const submitBtn = document.querySelector('[data-testid="baseButton-secondary"]');
                                if (submitBtn) {{
                                    setTimeout(() => submitBtn.click(), 500);
                                }}
                            }}
                        }}
                    }}, 100);
                }};

                recognition.onerror = function(event) {{
                    status.innerHTML = '❌ Error: ' + event.error + ' - Please try again';
                }};

                recognition.onend = function() {{
                    isListening = false;
                    btn.innerHTML = '🎤 Start Voice Input';
                    btn.style.background = 'linear-gradient(45deg, {PRIMARY_COLOR}, {SECONDARY_COLOR})';
                    status.innerHTML = 'Voice recognition stopped. Click to try again.';
                }};

                recognition.start();
            }} else {{
                status.innerHTML = '❌ Voice recognition not supported in this browser. Try Chrome or Edge.';
            }}
        }}

        function stopVoiceRecognition(btn, status) {{
            if (recognition) {{
                recognition.stop();
            }}
        }}
        </script>
        """
        return voice_html
    
    def create_text_to_speech_component(self, text_to_speak):
        """Create a simple Python-based text-to-speech component."""
        if not text_to_speak:
            return
            
        # Clean text for speech (remove markdown, emojis, etc.)
        clean_text = re.sub(r'[*_`#]', '', text_to_speak)  # Remove markdown
        clean_text = re.sub(r':[a-z_]+:', '', clean_text)  # Remove emoji codes like :smile:
        clean_text = remove_emojis(clean_text)  # Remove Unicode emojis
        clean_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', clean_text)  # Remove URLs
        clean_text = clean_text.strip()
        
        if len(clean_text) > 1000:  # Limit length for TTS
            clean_text = clean_text[:1000] + "..."
            
        # Show the new audio player
        if GTTS_AVAILABLE:
            st.subheader("🔊 Listen to Response")
            
            # Generate and display audio player
            try:
                tts = gTTS(text=clean_text, lang='en', slow=False)
                audio_buffer = BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                
                st.audio(audio_buffer.getvalue(), format='audio/mp3')
                
            except Exception as e:
                st.error(f"❌ Audio generation failed: {e}")
                st.info("💡 Try: `pip install gtts` to enable text-to-speech")
        else:
            st.warning("🔊 **Audio feature requires installation**: `pip install gtts`")
            
        return  # No HTML return needed
            
        tts_html = f"""
        <div style="text-align: center; margin: 1rem 0;">
            <button onclick="speakText()" 
                    style="background: linear-gradient(45deg, {SECONDARY_COLOR}, {PRIMARY_COLOR}); 
                           border: none; padding: 0.8rem 1.5rem; border-radius: 8px; color: white; 
                           font-size: 1rem; cursor: pointer;">
                🔊 Listen to Response
            </button>
            <button onclick="stopSpeech()" 
                    style="background: linear-gradient(45deg, #ff6b6b, #ee5a52); 
                           border: none; padding: 0.8rem 1.5rem; border-radius: 8px; color: white; 
                           font-size: 1rem; cursor: pointer; margin-left: 0.5rem;">
                ⏹️ Stop
            </button>
            <button onclick="testAudio()" 
                    style="background: linear-gradient(45deg, #fbbf24, #f59e0b); 
                           border: none; padding: 0.8rem 1.5rem; border-radius: 8px; color: white; 
                           font-size: 1rem; cursor: pointer; margin-left: 0.5rem;">
                🔊 Test Audio
            </button>
        </div>

        <script>
        let currentSpeech = null;

        // Make functions global to fix Streamlit scope issues
        window.speakText = function() {{
            console.log('speakText function called');
            if ('speechSynthesis' in window) {{
                console.log('speechSynthesis is supported');
                // Stop any ongoing speech
                window.stopSpeech();
                
                const text = `{clean_text}`;
                console.log('Text to speak:', text.substring(0, 100) + '...');
                currentSpeech = new SpeechSynthesisUtterance(text);
                
                // Configure voice
                currentSpeech.rate = 0.9;
                currentSpeech.pitch = 1;
                currentSpeech.volume = 0.8;
                
                // Set up event handlers first
                currentSpeech.onstart = function() {{
                    console.log('Speech started');
                }};
                
                currentSpeech.onend = function() {{
                    console.log('Speech ended');
                    currentSpeech = null;
                }};
                
                currentSpeech.onerror = function(event) {{
                    console.error('Speech error:', event.error);
                    alert('Speech error: ' + event.error);
                    currentSpeech = null;
                }};

                // Try to use a good voice (with voice loading fix)
                let voices = speechSynthesis.getVoices();
                
                function setVoiceAndSpeak() {{
                    console.log('Available voices:', voices.length);
                    const preferredVoices = voices.filter(voice => 
                        voice.lang.startsWith('en') && 
                        (voice.name.includes('Google') || voice.name.includes('Microsoft') || voice.name.includes('Natural'))
                    );
                    
                    if (preferredVoices.length > 0) {{
                        currentSpeech.voice = preferredVoices[0];
                        console.log('Using voice:', preferredVoices[0].name);
                    }} else if (voices.length > 0) {{
                        currentSpeech.voice = voices[0];
                        console.log('Using first available voice:', voices[0].name);
                    }} else {{
                        console.log('No voices available, using default');
                    }}
                    
                    // Actually start speaking
                    try {{
                        speechSynthesis.speak(currentSpeech);
                        console.log('speechSynthesis.speak() called with voice:', currentSpeech.voice ? currentSpeech.voice.name : 'default');
                    }} catch (error) {{
                        console.error('Error calling speechSynthesis.speak():', error);
                        alert('Error starting speech: ' + error.message);
                    }}
                }}
                
                // If no voices loaded, wait and try again
                if (voices.length === 0) {{
                    console.log('No voices loaded yet, waiting...');
                    speechSynthesis.onvoiceschanged = function() {{
                        voices = speechSynthesis.getVoices();
                        console.log('Voices loaded after wait:', voices.length);
                        setVoiceAndSpeak();
                    }};
                }} else {{
                    setVoiceAndSpeak();
                }}
            }} else {{
                console.error('speechSynthesis not supported');
                alert('Text-to-speech not supported in this browser. Try Chrome, Firefox, or Edge.');
            }}
        }}

        window.stopSpeech = function() {{
            if (speechSynthesis.speaking || currentSpeech) {{
                speechSynthesis.cancel();
                currentSpeech = null;
            }}
        }}

        window.testAudio = function() {{
            console.log('Testing audio...');
            if ('speechSynthesis' in window) {{
                window.stopSpeech();
                const testText = "Audio test. If you can hear this, your speakers and text to speech are working correctly.";
                const testSpeech = new SpeechSynthesisUtterance(testText);
                testSpeech.rate = 1.0;
                testSpeech.pitch = 1.0;
                testSpeech.volume = 1.0;
                
                testSpeech.onstart = function() {{
                    console.log('Test speech started');
                }};
                
                testSpeech.onend = function() {{
                    console.log('Test speech ended');
                }};
                
                testSpeech.onerror = function(event) {{
                    console.error('Test speech error:', event.error);
                    alert('Test speech error: ' + event.error);
                }};
                
                speechSynthesis.speak(testSpeech);
                console.log('Test speech initiated');
            }} else {{
                alert('Speech synthesis not supported');
            }}
        }}
        </script>
        """
        return tts_html
    
    def create_voice_commands_help(self):
        """Create a help section for voice commands."""
        help_html = f"""
        <div style="background: linear-gradient(135deg, {BG_GRADIENT_COLOR_2}, {BG_GRADIENT_COLOR_3}); 
                    padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
            <h4 style="color: {PRIMARY_COLOR}; margin-bottom: 1rem;">🎯 Voice Commands Guide</h4>
            <div style="color: {TEXT_COLOR}; font-size: 0.95rem; line-height: 1.6;">
                <strong>💰 Price Queries:</strong><br>
                • "What's the Bitcoin price?"<br>
                • "Show me Ethereum value"<br>
                • "How much is BTC trading at?"<br><br>
                
                <strong>📊 Charts:</strong><br>
                • "Show Bitcoin chart"<br>
                • "Display Ethereum graph"<br>
                • "Visualize Solana price"<br><br>
                
                <strong>⚖️ Comparisons:</strong><br>
                • "Compare Bitcoin and Ethereum"<br>
                • "Bitcoin versus Dogecoin"<br>
                • "Show BTC vs ETH"<br><br>
                
                <strong>🔥 Trending:</strong><br>
                • "What's trending?"<br>
                • "Show popular coins"<br>
                • "Top cryptocurrencies"<br><br>
                
                <strong>📚 Information:</strong><br>
                • "Explain Bitcoin"<br>
                • "What is blockchain?"<br>
                • "Tell me about DeFi"
            </div>
        </div>
        """
        return help_html
    
    def create_voice_status_component(self):
        """Create a compact voice status indicator."""
        status_html = f"""
        <div id="voice-status-compact" style="
            position: fixed; top: 20px; right: 20px; z-index: 1000;
            background: {SURFACE_COLOR}; border: 2px solid {PRIMARY_COLOR};
            border-radius: 25px; padding: 0.5rem 1rem; display: none;
            backdrop-filter: blur(10px); box-shadow: 0 4px 20px rgba(0,255,136,0.3);
            font-family: 'Orbitron', monospace; font-size: 0.85rem; color: {TEXT_COLOR};
        ">
            <span id="voice-indicator">🎤</span>
            <span id="voice-message">Ready</span>
        </div>
        
        <script>
        function updateVoiceStatus(message, icon = '🎤', show = true) {{
            const statusDiv = document.getElementById('voice-status-compact');
            const indicatorSpan = document.getElementById('voice-indicator');
            const messageSpan = document.getElementById('voice-message');
            
            if (statusDiv && indicatorSpan && messageSpan) {{
                indicatorSpan.textContent = icon;
                messageSpan.textContent = message;
                statusDiv.style.display = show ? 'block' : 'none';
                
                if (show) {{
                    setTimeout(() => {{
                        statusDiv.style.display = 'none';
                    }}, 3000);
                }}
            }}
        }}
        
        // Hook into the voice recognition events
        if (typeof window.originalStartVoiceRecognition === 'undefined') {{
            window.originalStartVoiceRecognition = window.startVoiceRecognition;
        }}
        </script>
        """
        return status_html
    
    def create_smart_voice_shortcuts(self):
        """Create smart voice shortcut buttons."""
        shortcuts_html = f"""
        <div style="background: linear-gradient(135deg, {BG_GRADIENT_COLOR_2}, {BG_GRADIENT_COLOR_3}); 
                    padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h4 style="color: {PRIMARY_COLOR}; margin-bottom: 1rem; text-align: center;">⚡ Quick Voice Actions</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                <button onclick="quickVoiceCommand('What\\'s the Bitcoin price?')" 
                        style="background: linear-gradient(45deg, {PRIMARY_COLOR}, {SECONDARY_COLOR}); 
                               border: none; padding: 0.5rem; border-radius: 8px; color: white; 
                               font-size: 0.9rem; cursor: pointer;">
                    💰 BTC Price
                </button>
                <button onclick="quickVoiceCommand('Show Ethereum chart')" 
                        style="background: linear-gradient(45deg, {SECONDARY_COLOR}, {ACCENT_COLOR}); 
                               border: none; padding: 0.5rem; border-radius: 8px; color: white; 
                               font-size: 0.9rem; cursor: pointer;">
                    📊 ETH Chart
                </button>
                <button onclick="quickVoiceCommand('What\\'s trending in crypto?')" 
                        style="background: linear-gradient(45deg, {ACCENT_COLOR}, #fbbf24); 
                               border: none; padding: 0.5rem; border-radius: 8px; color: white; 
                               font-size: 0.9rem; cursor: pointer;">
                    🔥 Trending
                </button>
                <button onclick="quickVoiceCommand('Compare Bitcoin and Ethereum')" 
                        style="background: linear-gradient(45deg, #fbbf24, {PRIMARY_COLOR}); 
                               border: none; padding: 0.5rem; border-radius: 8px; color: white; 
                               font-size: 0.9rem; cursor: pointer;">
                    ⚖️ Compare
                </button>
            </div>
        </div>
        
        <script>
        function quickVoiceCommand(command) {{
            updateVoiceStatus('Executing: ' + command, '⚡', true);
            
            const textArea = document.querySelector('[data-testid="stTextArea"] textarea');
            if (textArea) {{
                textArea.value = command;
                textArea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                
                // Auto-submit
                setTimeout(() => {{
                    const submitBtn = document.querySelector('[data-testid="baseButton-secondary"]');
                    if (submitBtn) {{
                        submitBtn.click();
                    }}
                }}, 200);
            }}
        }}
        </script>
        """
        return shortcuts_html

# --- Chat History Management Functions ---
class ChatHistoryManager:
    def __init__(self):
        self.history_dir = os.path.join(os.getcwd(), "chat_history")
        self.ensure_history_dir()
    
    def ensure_history_dir(self):
        """Create chat history directory if it doesn't exist."""
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
    
    def save_chat_session(self, messages: List[Dict], session_name: str = None) -> str:
        """Save a chat session to file."""
        if not messages:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not session_name:
            # Generate automatic name from first user message or timestamp
            first_user_msg = next((msg['content'] for msg in messages if msg['role'] == 'user'), "Chat")
            session_name = f"{first_user_msg[:30]}..." if len(first_user_msg) > 30 else first_user_msg
        
        # Clean session name for filename
        clean_name = "".join(c for c in session_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{timestamp}_{clean_name}.json"
        
        chat_data = {
            "timestamp": timestamp,
            "session_name": session_name,
            "messages": messages,
            "created_at": datetime.now().isoformat()
        }
        
        filepath = os.path.join(self.history_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(chat_data, f, indent=2, ensure_ascii=False)
            return filepath
        except Exception as e:
            st.error(f"Failed to save chat: {e}")
            return None
    
    def load_chat_sessions(self) -> List[Dict]:
        """Load all chat sessions from files."""
        sessions = []
        try:
            for filename in os.listdir(self.history_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.history_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                        session_data['filepath'] = filepath
                        sessions.append(session_data)
        except Exception as e:
            st.error(f"Failed to load chat history: {e}")
        
        # Sort by creation time (newest first)
        sessions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return sessions
    
    def delete_chat_session(self, filepath: str) -> bool:
        """Delete a specific chat session."""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
        except Exception as e:
            st.error(f"Failed to delete chat: {e}")
        return False
    
    def search_chat_history(self, query: str, sessions: List[Dict]) -> List[Dict]:
        """Search through chat history for specific content."""
        if not query:
            return sessions
        
        query_lower = query.lower()
        matching_sessions = []
        
        for session in sessions:
            # Search in session name
            if query_lower in session.get('session_name', '').lower():
                matching_sessions.append(session)
                continue
            
            # Search in message content
            for message in session.get('messages', []):
                if query_lower in message.get('content', '').lower():
                    matching_sessions.append(session)
                    break
        
        return matching_sessions
    
    def export_chat_as_text(self, session_data: Dict) -> str:
        """Export a chat session as formatted text."""
        text = f"Chat Session: {session_data.get('session_name', 'Untitled')}\n"
        text += f"Date: {session_data.get('created_at', 'Unknown')}\n"
        text += "=" * 50 + "\n\n"
        
        for message in session_data.get('messages', []):
            role = "You" if message['role'] == 'user' else "Kryptonic AI"
            text += f"{role}: {message['content']}\n\n"
        
        return text
    
    def process_voice_command(self, voice_text):
        """Process voice input and identify command type with smart suggestions."""
        if not voice_text:
            return None, None
            
        voice_text_lower = voice_text.lower()
        
        # Extract cryptocurrency names
        detected_coins = []
        coin_aliases = {
            'btc': 'bitcoin', 'bitcoin': 'bitcoin',
            'eth': 'ethereum', 'ethereum': 'ethereum', 
            'bnb': 'binancecoin', 'binance coin': 'binancecoin',
            'ada': 'cardano', 'cardano': 'cardano',
            'sol': 'solana', 'solana': 'solana',
            'dot': 'polkadot', 'polkadot': 'polkadot',
            'doge': 'dogecoin', 'dogecoin': 'dogecoin',
            'link': 'chainlink', 'chainlink': 'chainlink'
        }
        
        for alias, coin_id in coin_aliases.items():
            if alias in voice_text_lower:
                detected_coins.append((alias.title(), coin_id))
        
        # Identify command type
        command_type = None
        for cmd, patterns in self.voice_commands.items():
            if any(pattern in voice_text_lower for pattern in patterns):
                command_type = cmd
                break
        
        # Generate smart response based on detected pattern
        suggestion = self.generate_voice_suggestion(command_type, detected_coins, voice_text)
        
        return command_type, detected_coins, suggestion
    
    def generate_voice_suggestion(self, command_type, detected_coins, original_text):
        """Generate smart suggestions for voice commands."""
        if not command_type and not detected_coins:
            return "💡 Try saying: 'What's the Bitcoin price?' or 'Show me Ethereum chart'"
        
        if command_type == 'price' and detected_coins:
            coin_name = detected_coins[0][0]
            return f"🎯 Getting {coin_name} price information..."
        
        if command_type == 'chart' and detected_coins:
            coin_name = detected_coins[0][0]
            return f"📊 Opening {coin_name} chart..."
        
        if command_type == 'trending':
            return "🔥 Loading trending cryptocurrencies..."
        
        if command_type == 'compare' and len(detected_coins) >= 2:
            coins = " vs ".join([coin[0] for coin in detected_coins[:2]])
            return f"⚖️ Comparing {coins}..."
        
        return "🎤 Processing your voice command..."
    
    def create_voice_input_component(self):
        """Create a voice input component using HTML5 Web Speech API."""
        voice_html = f"""
        <div style="background: linear-gradient(135deg, {BG_GRADIENT_COLOR_1}, {BG_GRADIENT_COLOR_2}); 
                    padding: 1.5rem; border-radius: 15px; border: 2px solid {PRIMARY_COLOR};
                    text-align: center; margin: 1rem 0;">
            <h3 style="color: {PRIMARY_COLOR}; margin-bottom: 1rem;">🎤 Voice Input</h3>
            
            <button id="voice-btn" onclick="toggleVoiceRecognition()" 
                    style="background: linear-gradient(45deg, {PRIMARY_COLOR}, {SECONDARY_COLOR}); 
                           border: none; padding: 1rem 2rem; border-radius: 10px; color: white; 
                           font-size: 1.1rem; cursor: pointer; margin: 0.5rem;">
                🎤 Start Voice Input
            </button>
            
            <div id="voice-status" style="color: {TEXT_COLOR}; margin-top: 1rem; font-size: 1rem;">
                Click the button and speak your crypto question
            </div>
            
            <div id="voice-result" style="background: rgba(0,255,136,0.1); padding: 1rem; 
                                         border-radius: 10px; margin-top: 1rem; color: {TEXT_COLOR}; 
                                         font-family: monospace; min-height: 2rem; display: none;">
            </div>
        </div>

        <script>
        let recognition;
        let isListening = false;

        function toggleVoiceRecognition() {{
            const btn = document.getElementById('voice-btn');
            const status = document.getElementById('voice-status');
            const result = document.getElementById('voice-result');
            
            if (!isListening) {{
                startVoiceRecognition(btn, status, result);
            }} else {{
                stopVoiceRecognition(btn, status);
            }}
        }}

        function startVoiceRecognition(btn, status, result) {{
            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {{
                recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
                recognition.continuous = false;
                recognition.interimResults = false;
                recognition.lang = 'en-US';

                recognition.onstart = function() {{
                    isListening = true;
                    btn.innerHTML = '🔴 Listening...';
                    btn.style.background = 'linear-gradient(45deg, #ff4444, #ff6666)';
                    status.innerHTML = '🎧 Listening... Speak now!';
                    result.style.display = 'none';
                }};

                recognition.onresult = function(event) {{
                    const transcript = event.results[0][0].transcript;
                    const confidence = event.results[0][0].confidence;
                    
                    let confidenceColor = confidence > 0.8 ? '{PRIMARY_COLOR}' : confidence > 0.6 ? '#fbbf24' : '#ff6b6b';
                    
                    result.innerHTML = `
                        <strong>You said:</strong> "${transcript}"<br>
                        <small style="color: ${confidenceColor};">Confidence: ${Math.round(confidence * 100)}%</small>
                    `;
                    result.style.display = 'block';
                    
                    // Send result to Streamlit with slight delay to ensure visibility
                    setTimeout(() => {{
                        const textArea = document.querySelector('[data-testid="stTextArea"] textarea');
                        if (textArea) {{
                            textArea.value = transcript;
                            textArea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            
                            // Auto-submit if confidence is high
                            if (confidence > 0.85) {{
                                const submitBtn = document.querySelector('[data-testid="baseButton-secondary"]');
                                if (submitBtn) {{
                                    setTimeout(() => submitBtn.click(), 500);
                                }}
                            }}
                        }}
                    }}, 100);
                }};

                recognition.onerror = function(event) {{
                    status.innerHTML = '❌ Error: ' + event.error + ' - Please try again';
                }};

                recognition.onend = function() {{
                    isListening = false;
                    btn.innerHTML = '🎤 Start Voice Input';
                    btn.style.background = 'linear-gradient(45deg, {PRIMARY_COLOR}, {SECONDARY_COLOR})';
                    status.innerHTML = 'Voice recognition stopped. Click to try again.';
                }};

                recognition.start();
            }} else {{
                status.innerHTML = '❌ Voice recognition not supported in this browser. Try Chrome or Edge.';
            }}
        }}

        function stopVoiceRecognition(btn, status) {{
            if (recognition) {{
                recognition.stop();
            }}
        }}
        </script>
        """
        return voice_html
    
    def create_text_to_speech_component(self, text_to_speak):
        """Create a simple Python-based text-to-speech component."""
        if not text_to_speak:
            return
            
        # Clean text for speech (remove markdown, emojis, etc.)
        clean_text = re.sub(r'[*_`#]', '', text_to_speak)  # Remove markdown
        clean_text = re.sub(r':[a-z_]+:', '', clean_text)  # Remove emoji codes like :smile:
        clean_text = remove_emojis(clean_text)  # Remove Unicode emojis
        clean_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', clean_text)  # Remove URLs
        clean_text = clean_text.strip()
        
        if len(clean_text) > 1000:  # Limit length for TTS
            clean_text = clean_text[:1000] + "..."
            
        # Show the new audio player
        if GTTS_AVAILABLE:
            st.subheader("🔊 Listen to Response")
            
            # Generate and display audio player
            try:
                tts = gTTS(text=clean_text, lang='en', slow=False)
                audio_buffer = BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                
                st.audio(audio_buffer.getvalue(), format='audio/mp3')
                
            except Exception as e:
                st.error(f"❌ Audio generation failed: {e}")
                st.info("💡 Try: `pip install gtts` to enable text-to-speech")
        else:
            st.warning("🔊 **Audio feature requires installation**: `pip install gtts`")
            
        return  # No HTML return needed
            
        tts_html = f"""
        <div style="text-align: center; margin: 1rem 0;">
            <button onclick="speakText()" 
                    style="background: linear-gradient(45deg, {SECONDARY_COLOR}, {PRIMARY_COLOR}); 
                           border: none; padding: 0.8rem 1.5rem; border-radius: 8px; color: white; 
                           font-size: 1rem; cursor: pointer;">
                🔊 Listen to Response
            </button>
            <button onclick="stopSpeech()" 
                    style="background: linear-gradient(45deg, #ff6b6b, #ee5a52); 
                           border: none; padding: 0.8rem 1.5rem; border-radius: 8px; color: white; 
                           font-size: 1rem; cursor: pointer; margin-left: 0.5rem;">
                ⏹️ Stop
            </button>
            <button onclick="testAudio()" 
                    style="background: linear-gradient(45deg, #fbbf24, #f59e0b); 
                           border: none; padding: 0.8rem 1.5rem; border-radius: 8px; color: white; 
                           font-size: 1rem; cursor: pointer; margin-left: 0.5rem;">
                🔊 Test Audio
            </button>
        </div>

        <script>
        let currentSpeech = null;

        // Make functions global to fix Streamlit scope issues
        window.speakText = function() {{
            console.log('speakText function called');
            if ('speechSynthesis' in window) {{
                console.log('speechSynthesis is supported');
                // Stop any ongoing speech
                window.stopSpeech();
                
                const text = `{clean_text}`;
                console.log('Text to speak:', text.substring(0, 100) + '...');
                currentSpeech = new SpeechSynthesisUtterance(text);
                
                // Configure voice
                currentSpeech.rate = 0.9;
                currentSpeech.pitch = 1;
                currentSpeech.volume = 0.8;
                
                // Set up event handlers first
                currentSpeech.onstart = function() {{
                    console.log('Speech started');
                }};
                
                currentSpeech.onend = function() {{
                    console.log('Speech ended');
                    currentSpeech = null;
                }};
                
                currentSpeech.onerror = function(event) {{
                    console.error('Speech error:', event.error);
                    alert('Speech error: ' + event.error);
                    currentSpeech = null;
                }};

                // Try to use a good voice (with voice loading fix)
                let voices = speechSynthesis.getVoices();
                
                function setVoiceAndSpeak() {{
                    console.log('Available voices:', voices.length);
                    const preferredVoices = voices.filter(voice => 
                        voice.lang.startsWith('en') && 
                        (voice.name.includes('Google') || voice.name.includes('Microsoft') || voice.name.includes('Natural'))
                    );
                    
                    if (preferredVoices.length > 0) {{
                        currentSpeech.voice = preferredVoices[0];
                        console.log('Using voice:', preferredVoices[0].name);
                    }} else if (voices.length > 0) {{
                        currentSpeech.voice = voices[0];
                        console.log('Using first available voice:', voices[0].name);
                    }} else {{
                        console.log('No voices available, using default');
                    }}
                    
                    // Actually start speaking
                    try {{
                        speechSynthesis.speak(currentSpeech);
                        console.log('speechSynthesis.speak() called with voice:', currentSpeech.voice ? currentSpeech.voice.name : 'default');
                    }} catch (error) {{
                        console.error('Error calling speechSynthesis.speak():', error);
                        alert('Error starting speech: ' + error.message);
                    }}
                }}
                
                // If no voices loaded, wait and try again
                if (voices.length === 0) {{
                    console.log('No voices loaded yet, waiting...');
                    speechSynthesis.onvoiceschanged = function() {{
                        voices = speechSynthesis.getVoices();
                        console.log('Voices loaded after wait:', voices.length);
                        setVoiceAndSpeak();
                    }};
                }} else {{
                    setVoiceAndSpeak();
                }}
            }} else {{
                console.error('speechSynthesis not supported');
                alert('Text-to-speech not supported in this browser. Try Chrome, Firefox, or Edge.');
            }}
        }}

        window.stopSpeech = function() {{
            if (speechSynthesis.speaking || currentSpeech) {{
                speechSynthesis.cancel();
                currentSpeech = null;
            }}
        }}

        window.testAudio = function() {{
            console.log('Testing audio...');
            if ('speechSynthesis' in window) {{
                window.stopSpeech();
                const testText = "Audio test. If you can hear this, your speakers and text to speech are working correctly.";
                const testSpeech = new SpeechSynthesisUtterance(testText);
                testSpeech.rate = 1.0;
                testSpeech.pitch = 1.0;
                testSpeech.volume = 1.0;
                
                testSpeech.onstart = function() {{
                    console.log('Test speech started');
                }};
                
                testSpeech.onend = function() {{
                    console.log('Test speech ended');
                }};
                
                testSpeech.onerror = function(event) {{
                    console.error('Test speech error:', event.error);
                    alert('Test speech error: ' + event.error);
                }};
                
                speechSynthesis.speak(testSpeech);
                console.log('Test speech initiated');
            }} else {{
                alert('Speech synthesis not supported');
            }}
        }}
        </script>
        """
        return tts_html
    
    def create_voice_commands_help(self):
        """Create a help section for voice commands."""
        help_html = f"""
        <div style="background: linear-gradient(135deg, {BG_GRADIENT_COLOR_2}, {BG_GRADIENT_COLOR_3}); 
                    padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
            <h4 style="color: {PRIMARY_COLOR}; margin-bottom: 1rem;">🎯 Voice Commands Guide</h4>
            <div style="color: {TEXT_COLOR}; font-size: 0.95rem; line-height: 1.6;">
                <strong>💰 Price Queries:</strong><br>
                • "What's the Bitcoin price?"<br>
                • "Show me Ethereum value"<br>
                • "How much is BTC trading at?"<br><br>
                
                <strong>📊 Charts:</strong><br>
                • "Show Bitcoin chart"<br>
                • "Display Ethereum graph"<br>
                • "Visualize Solana price"<br><br>
                
                <strong>⚖️ Comparisons:</strong><br>
                • "Compare Bitcoin and Ethereum"<br>
                • "Bitcoin versus Dogecoin"<br>
                • "Show BTC vs ETH"<br><br>
                
                <strong>🔥 Trending:</strong><br>
                • "What's trending?"<br>
                • "Show popular coins"<br>
                • "Top cryptocurrencies"<br><br>
                
                <strong>📚 Information:</strong><br>
                • "Explain Bitcoin"<br>
                • "What is blockchain?"<br>
                • "Tell me about DeFi"
            </div>
        </div>
        """
        return help_html
    
    def create_voice_status_component(self):
        """Create a compact voice status indicator."""
        status_html = f"""
        <div id="voice-status-compact" style="
            position: fixed; top: 20px; right: 20px; z-index: 1000;
            background: {SURFACE_COLOR}; border: 2px solid {PRIMARY_COLOR};
            border-radius: 25px; padding: 0.5rem 1rem; display: none;
            backdrop-filter: blur(10px); box-shadow: 0 4px 20px rgba(0,255,136,0.3);
            font-family: 'Orbitron', monospace; font-size: 0.85rem; color: {TEXT_COLOR};
        ">
            <span id="voice-indicator">🎤</span>
            <span id="voice-message">Ready</span>
        </div>
        
        <script>
        function updateVoiceStatus(message, icon = '🎤', show = true) {{
            const statusDiv = document.getElementById('voice-status-compact');
            const indicatorSpan = document.getElementById('voice-indicator');
            const messageSpan = document.getElementById('voice-message');
            
            if (statusDiv && indicatorSpan && messageSpan) {{
                indicatorSpan.textContent = icon;
                messageSpan.textContent = message;
                statusDiv.style.display = show ? 'block' : 'none';
                
                if (show) {{
                    setTimeout(() => {{
                        statusDiv.style.display = 'none';
                    }}, 3000);
                }}
            }}
        }}
        
        // Hook into the voice recognition events
        if (typeof window.originalStartVoiceRecognition === 'undefined') {{
            window.originalStartVoiceRecognition = window.startVoiceRecognition;
        }}
        </script>
        """
        return status_html
    
    def create_smart_voice_shortcuts(self):
        """Create smart voice shortcut buttons."""
        shortcuts_html = f"""
        <div style="background: linear-gradient(135deg, {BG_GRADIENT_COLOR_2}, {BG_GRADIENT_COLOR_3}); 
                    padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <h4 style="color: {PRIMARY_COLOR}; margin-bottom: 1rem; text-align: center;">⚡ Quick Voice Actions</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                <button onclick="quickVoiceCommand('What\\'s the Bitcoin price?')" 
                        style="background: linear-gradient(45deg, {PRIMARY_COLOR}, {SECONDARY_COLOR}); 
                               border: none; padding: 0.5rem; border-radius: 8px; color: white; 
                               font-size: 0.9rem; cursor: pointer;">
                    💰 BTC Price
                </button>
                <button onclick="quickVoiceCommand('Show Ethereum chart')" 
                        style="background: linear-gradient(45deg, {SECONDARY_COLOR}, {ACCENT_COLOR}); 
                               border: none; padding: 0.5rem; border-radius: 8px; color: white; 
                               font-size: 0.9rem; cursor: pointer;">
                    📊 ETH Chart
                </button>
                <button onclick="quickVoiceCommand('What\\'s trending in crypto?')" 
                        style="background: linear-gradient(45deg, {ACCENT_COLOR}, #fbbf24); 
                               border: none; padding: 0.5rem; border-radius: 8px; color: white; 
                               font-size: 0.9rem; cursor: pointer;">
                    🔥 Trending
                </button>
                <button onclick="quickVoiceCommand('Compare Bitcoin and Ethereum')" 
                        style="background: linear-gradient(45deg, #fbbf24, {PRIMARY_COLOR}); 
                               border: none; padding: 0.5rem; border-radius: 8px; color: white; 
                               font-size: 0.9rem; cursor: pointer;">
                    ⚖️ Compare
                </button>
            </div>
        </div>
        
        <script>
        function quickVoiceCommand(command) {{
            updateVoiceStatus('Executing: ' + command, '⚡', true);
            
            const textArea = document.querySelector('[data-testid="stTextArea"] textarea');
            if (textArea) {{
                textArea.value = command;
                textArea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                
                // Auto-submit
                setTimeout(() => {{
                    const submitBtn = document.querySelector('[data-testid="baseButton-secondary"]');
                    if (submitBtn) {{
                        submitBtn.click();
                    }}
                }}, 200);
            }}
        }}
        </script>
        """
        return shortcuts_html

# --- Fallback Voice Features (Alternative Implementation) ---
# Note: This HTML-based fallback was causing display issues, replaced with Streamlit components
# def create_fallback_voice_features():
#     """Create a simplified voice interface as fallback when main voice features fail."""
#     # Replaced with pure Streamlit components to avoid HTML rendering issues

class GamificationManager:
    """Manages user achievements, challenges, and gamification features."""
    
    def __init__(self):
        self.achievements_file = os.path.join("c:\\Users\\Verona\\Desktop\\eccu camp", "chat_history", "achievements.json")
        self.user_stats_file = os.path.join("c:\\Users\\Verona\\Desktop\\eccu camp", "chat_history", "user_stats.json")
        self.ensure_files_exist()
    
    def ensure_files_exist(self):
        """Ensure gamification files exist."""
        os.makedirs(os.path.dirname(self.achievements_file), exist_ok=True)
        
        # Initialize achievements file
        if not os.path.exists(self.achievements_file):
            initial_achievements = {
                "unlocked": [],
                "progress": {}
            }
            with open(self.achievements_file, 'w') as f:
                json.dump(initial_achievements, f, indent=2)
        
        # Initialize user stats file
        if not os.path.exists(self.user_stats_file):
            initial_stats = {
                "total_xp": 0,
                "level": 1,
                "messages_sent": 0,
                "days_active": 0,
                "last_active": None,
                "current_streak": 0,
                "longest_streak": 0,
                "coins_learned": [],
                "topics_explored": [],
                "daily_challenge_completed": False,
                "last_challenge_date": None,
                "total_challenges_completed": 0
            }
            with open(self.user_stats_file, 'w') as f:
                json.dump(initial_stats, f, indent=2)
    
    def get_achievements_definition(self):
        """Define all available achievements."""
        return {
            "first_chat": {
                "name": "🎬 First Contact", 
                "description": "Started your first conversation with Kryptonic",
                "xp_reward": 50,
                "icon": "🎬"
            },
            "crypto_curious": {
                "name": "🔍 Crypto Curious", 
                "description": "Asked 10 crypto-related questions",
                "xp_reward": 100,
                "icon": "🔍",
                "requirement": 10
            },
            "price_checker": {
                "name": "💰 Price Hunter", 
                "description": "Checked cryptocurrency prices 5 times",
                "xp_reward": 75,
                "icon": "💰",
                "requirement": 5
            },
            "learning_streak": {
                "name": "🔥 Learning Streak", 
                "description": "Used the app for 7 consecutive days",
                "xp_reward": 200,
                "icon": "🔥",
                "requirement": 7
            },
            "crypto_explorer": {
                "name": "🌟 Crypto Explorer", 
                "description": "Learned about 10 different cryptocurrencies",
                "xp_reward": 150,
                "icon": "🌟",
                "requirement": 10
            },
            "market_analyst": {
                "name": "📊 Market Analyst", 
                "description": "Viewed market data 20 times",
                "xp_reward": 125,
                "icon": "📊",
                "requirement": 20
            },
            "challenge_master": {
                "name": "🏆 Challenge Master", 
                "description": "Completed 30 daily challenges",
                "xp_reward": 300,
                "icon": "🏆",
                "requirement": 30
            },
            "crypto_veteran": {
                "name": "💎 Crypto Veteran", 
                "description": "Reached level 10",
                "xp_reward": 500,
                "icon": "💎",
                "requirement": 10
            },
            "social_learner": {
                "name": "🤝 Social Learner", 
                "description": "Shared 5 AI responses or insights",
                "xp_reward": 100,
                "icon": "🤝",
                "requirement": 5
            },
            "night_owl": {
                "name": "🦉 Night Owl", 
                "description": "Used the app after midnight 10 times",
                "xp_reward": 75,
                "icon": "🦉",
                "requirement": 10
            }
        }
    
    def load_user_stats(self):
        """Load user statistics."""
        try:
            with open(self.user_stats_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file is missing or corrupt, re-initialize it
            self.ensure_files_exist()
            with open(self.user_stats_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Failed to load user stats: {e}")
            # Return a default structure on other errors
            return {
                "total_xp": 0, "level": 1, "messages_sent": 0, "days_active": 0,
                "last_active": None, "current_streak": 0, "longest_streak": 0,
                "coins_learned": [], "topics_explored": [], "daily_challenge_completed": False,
                "last_challenge_date": None, "total_challenges_completed": 0
            }
    
    def save_user_stats(self, stats):
        """Save user statistics."""
        try:
            with open(self.user_stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            st.error(f"Failed to save user stats: {e}")
    
    def load_achievements(self):
        """Load user achievements."""
        try:
            with open(self.achievements_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {"unlocked": [], "progress": {}}
    
    def save_achievements(self, achievements):
        """Save user achievements."""
        try:
            with open(self.achievements_file, 'w') as f:
                json.dump(achievements, f, indent=2)
        except Exception as e:
            st.error(f"Failed to save achievements: {e}")
    
    def add_xp(self, amount, reason=""):
        """Add XP to user and check for level up."""
        stats = self.load_user_stats()
        stats["total_xp"] += amount
        
        # Calculate level (every 100 XP = 1 level)
        new_level = (stats["total_xp"] // 100) + 1
        level_up = new_level > stats["level"]
        stats["level"] = new_level
        
        self.save_user_stats(stats)
        
        if level_up:
            st.balloons()
            st.success(f"🎉 LEVEL UP! You're now level {new_level}! (+{amount} XP)")
            self.check_achievement("crypto_veteran", stats["level"])
        else:
            st.info(f"✨ +{amount} XP earned! {reason}")
        
        return level_up
    
    def update_daily_activity(self):
        """Update daily activity and streaks."""
        stats = self.load_user_stats()
        today = date.today().isoformat()
        
        if stats["last_active"] != today:
            # Check if this continues a streak
            if stats["last_active"]:
                last_date = datetime.fromisoformat(stats["last_active"]).date()
                days_diff = (date.today() - last_date).days
                
                if days_diff == 1:
                    # Continuing streak
                    stats["current_streak"] += 1
                elif days_diff > 1:
                    # Streak broken
                    stats["current_streak"] = 1
            else:
                # First day
                stats["current_streak"] = 1
            
            stats["last_active"] = today
            stats["days_active"] += 1
            
            # Update longest streak
            if stats["current_streak"] > stats["longest_streak"]:
                stats["longest_streak"] = stats["current_streak"]
            
            # Check streak achievements
            if stats["current_streak"] >= 7:
                self.check_achievement("learning_streak", stats["current_streak"])
            
            self.save_user_stats(stats)
            self.add_xp(10, "Daily login bonus!")
        
        return stats
    
    def record_message(self, message_content):
        """Record a user message for gamification tracking."""
        stats = self.load_user_stats()
        stats["messages_sent"] += 1
        
        # Check for first message
        if stats["messages_sent"] == 1:
            self.unlock_achievement("first_chat")
        
        # Extract crypto mentions
        crypto_keywords = ["bitcoin", "ethereum", "btc", "eth", "crypto", "blockchain", "defi", "nft"]
        message_lower = message_content.lower()
        
        for keyword in crypto_keywords:
            if keyword in message_lower and keyword not in stats["coins_learned"]:
                stats["coins_learned"].append(keyword)
        
        self.save_user_stats(stats)
        
        # Check achievements
        self.check_achievement("crypto_curious", stats["messages_sent"])
        self.check_achievement("crypto_explorer", len(stats["coins_learned"]))
    
    def record_price_check(self, coin_name):
        """Record when user checks a price."""
        stats = self.load_user_stats()
        if "price_checks" not in stats:
            stats["price_checks"] = 0
        stats["price_checks"] += 1
        
        # Add coin to learned list
        if coin_name not in stats["coins_learned"]:
            stats["coins_learned"].append(coin_name)
        
        self.save_user_stats(stats)
        self.check_achievement("price_checker", stats["price_checks"])
        self.check_achievement("crypto_explorer", len(stats["coins_learned"]))
    
    def record_market_view(self):
        """Record when user views market data."""
        stats = self.load_user_stats()
        if "market_views" not in stats:
            stats["market_views"] = 0
        stats["market_views"] += 1
        self.save_user_stats(stats)
        self.check_achievement("market_analyst", stats["market_views"])
    
    def check_achievement(self, achievement_key, current_value):
        """Check if an achievement should be unlocked."""
        achievements_def = self.get_achievements_definition()
        achievements = self.load_achievements()
        
        if achievement_key in achievements_def and achievement_key not in achievements["unlocked"]:
            achievement = achievements_def[achievement_key]
            requirement = achievement.get("requirement", 1)
            
            if current_value >= requirement:
                self.unlock_achievement(achievement_key)
    
    def unlock_achievement(self, achievement_key):
        """Unlock a specific achievement."""
        achievements = self.load_achievements()
        achievements_def = self.get_achievements_definition()
        
        if achievement_key not in achievements["unlocked"] and achievement_key in achievements_def:
            achievements["unlocked"].append(achievement_key)
            self.save_achievements(achievements)
            
            achievement = achievements_def[achievement_key]
            
            # Show celebration first
            st.balloons()
            
            # Create a special achievement notification
            st.success(f"""
🏆 **ACHIEVEMENT UNLOCKED!** 🏆

{achievement['icon']} **{achievement['name']}**

*{achievement['description']}*

**Reward: +{achievement['xp_reward']} XP!**
            """)
            
            # Add XP without showing the regular XP message since we already showed the achievement
            stats = self.load_user_stats()
            stats["total_xp"] += achievement["xp_reward"]
            
            # Calculate level (every 100 XP = 1 level)
            new_level = (stats["total_xp"] // 100) + 1
            level_up = new_level > stats["level"]
            stats["level"] = new_level
            
            self.save_user_stats(stats)
            
            if level_up:
                st.balloons()
                st.success(f"🎉 **DOUBLE CELEBRATION!** 🎉\n\nLevel Up! You're now **Level {new_level}**!")
                self.check_achievement("crypto_veteran", stats["level"])
            
            return True
        return False
    
    def get_daily_challenge(self):
        """Get today's daily challenge."""
        challenges = [
            {
                "question": "What does 'HODL' mean in cryptocurrency?",
                "options": ["Hold On for Dear Life", "High-Order Digital Ledger", "Hybrid Online Data Link", "Hashed Output Distributed Ledger"],
                "correct": 0,
                "explanation": "HODL originated from a typo of 'hold' and now means 'Hold On for Dear Life' - a strategy of holding crypto long-term.",
                "xp_reward": 25
            },
            {
                "question": "What is the maximum supply of Bitcoin?",
                "options": ["21 million", "100 million", "No limit", "50 million"],
                "correct": 0,
                "explanation": "Bitcoin has a hard cap of 21 million coins, making it deflationary by design.",
                "xp_reward": 30
            },
            {
                "question": "What does DeFi stand for?",
                "options": ["Digital Finance", "Decentralized Finance", "Dynamic Finance", "Distributed Finance"],
                "correct": 1,
                "explanation": "DeFi means Decentralized Finance - financial services built on blockchain without traditional intermediaries.",
                "xp_reward": 25
            },
            {
                "question": "Which blockchain is Ethereum's main competitor for smart contracts?",
                "options": ["Bitcoin", "Binance Smart Chain", "Litecoin", "Dogecoin"],
                "correct": 1,
                "explanation": "Binance Smart Chain is a major competitor to Ethereum for smart contract functionality.",
                "xp_reward": 35
            },
            {
                "question": "What is a crypto wallet's private key?",
                "options": ["Your password", "A secret code that controls your crypto", "Your username", "A public address"],
                "correct": 1,
                "explanation": "A private key is a secret code that gives you control over your cryptocurrency. Never share it!",
                "xp_reward": 40
            }
        ]
        
        # Use date as seed for consistent daily challenge
        today = date.today()
        random.seed(today.toordinal())
        return random.choice(challenges)
    
    def complete_daily_challenge(self, is_correct):
        """Complete today's daily challenge."""
        stats = self.load_user_stats()
        today = date.today().isoformat()
        
        if stats.get("last_challenge_date") != today:
            stats["last_challenge_date"] = today
            stats["daily_challenge_completed"] = True
            stats["total_challenges_completed"] += 1
            
            xp_reward = 25 if is_correct else 10
            self.add_xp(xp_reward, f"Daily challenge {'completed correctly' if is_correct else 'attempted'}!")
            
            self.save_user_stats(stats)
            self.check_achievement("challenge_master", stats["total_challenges_completed"])
            
            return True
        return False

def get_theme_css(is_dark_mode, animations_enabled, reduce_motion=False):
    """Generate CSS based on theme and animation preferences."""
    
    if is_dark_mode:
        bg_gradient = f"linear-gradient(135deg, {BG_GRADIENT_COLOR_1}, {BG_GRADIENT_COLOR_2}, {BG_GRADIENT_COLOR_3}, {BG_GRADIENT_COLOR_4}, {BG_GRADIENT_COLOR_5})"
        primary_color_chat = PRIMARY_COLOR
        secondary_color_chat = SECONDARY_COLOR
        accent_color_chat = ACCENT_COLOR
        text_color_chat = TEXT_COLOR
        surface_color_chat = SURFACE_COLOR
        border_color_chat = BORDER_COLOR
        button_text_color_chat = "#000000"
        sidebar_header_color = primary_color_chat
        main_text_color = TEXT_COLOR
        sidebar_text_color = TEXT_COLOR
        toggle_label_color = TEXT_COLOR
    else:
        bg_gradient = "linear-gradient(135deg, #f0f2f6, #e9eef2)"
        primary_color_chat = "#059669"
        secondary_color_chat = "#0284c7"
        accent_color_chat = "#dc2626"
        text_color_chat = "#000000"
        surface_color_chat = "rgba(255, 255, 255, 0.9)"
        border_color_chat = "rgba(5, 150, 105, 0.5)"
        button_text_color_chat = "#ffffff"
        sidebar_header_color = "#000000"
        main_text_color = "#000000"
        sidebar_text_color = "#000000"
        toggle_label_color = "#000000"
    
    animation_css = ""
    if animations_enabled and not reduce_motion:
        animation_css = """
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes borderGlow {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        @keyframes dataPulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.3; transform: scale(0.8); }
        }
        @keyframes cursorBlink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0; }
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.3; }
            50% { transform: scale(1.2); opacity: 0.5; }
        }
        @keyframes float {
            0%, 100% { transform: translateY(0px) translateX(0px); }
            25% { transform: translateY(-15px) translateX(5px); }
            50% { transform: translateY(-5px) translateX(-5px); }
            75% { transform: translateY(-20px) translateX(8px); }
        }
        @keyframes glitchEffect {
            0% { transform: translateX(0); }
            20% { transform: translateX(-2px); }
            40% { transform: translateX(2px); }
            60% { transform: translateX(-1px); }
            80% { transform: translateX(1px); }
            100% { transform: translateX(0); }
        }
        .main-header { animation: gradientShift 3s ease infinite; }
        .ai-message::before { animation: borderGlow 2s ease-in-out infinite; }
        .ai-message .data-stream { animation: dataPulse 1.5s ease-in-out infinite; }
        .ai-message .data-stream::before { animation: dataPulse 1.8s ease-in-out infinite; }
        .ai-message .data-stream::after { animation: dataPulse 2.1s ease-in-out infinite; }
        .user-message .terminal-cursor { animation: cursorBlink 1s infinite; }
        .glow-text { animation: pulse 2s infinite; }
        .user-message:hover { animation: glitchEffect 0.3s ease-in-out; }
        .glow-orb { animation: pulse 4s ease-in-out infinite; }
        .crystal-placeholder { animation: float 8s ease-in-out infinite; }
        """
    elif reduce_motion:
        # Provide safe, minimal animations for users who need reduced motion
        animation_css = """
        @keyframes gentleFade {
            0% { opacity: 0.95; }
            100% { opacity: 1; }
        }
        /* Respect prefers-reduced-motion */
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
                scroll-behavior: auto !important;
            }
        }
        /* Disable all potentially problematic animations */
        .main-header { animation: none; }
        .ai-message::before { background: linear-gradient(45deg, """ + secondary_color_chat + """, """ + accent_color_chat + """, """ + primary_color_chat + """); animation: none; }
        .ai-message .data-stream, .ai-message .data-stream::before, .ai-message .data-stream::after { animation: none; }
        .user-message .terminal-cursor { animation: none; opacity: 1; }
        .glow-text { animation: none; }
        .user-message:hover, .glow-orb, .crystal-placeholder { animation: none; }
        .hero-title .highlight { animation: none !important; }
        /* Gentle hover effects only */
        .ai-message:hover { transform: none; transition: none; }
        """
    else:
        animation_css = """
        .main-header { animation: none; }
        .ai-message::before { animation: none; }
        .ai-message .data-stream { animation: none; }
        .ai-message .data-stream::before { animation: none; }
        .ai-message .data-stream::after { animation: none; }
        .user-message .terminal-cursor { animation: none; }
        .glow-text { animation: none; }
        .user-message:hover { animation: none; }
        .glow-orb { animation: none; }
        .crystal-placeholder { animation: none; }
        """
    
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    body {{
        font-family: 'Inter', sans-serif;
        background: {bg_gradient};
        color: {main_text_color};
        overflow-x: hidden;
        min-height: 100vh;
        position: relative;
    }}
    .stApp {{
        background: {bg_gradient};
        color: {main_text_color};
        font-family: 'Inter', sans-serif;
    }}
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: 1200px;
        margin: 0 auto;
    }}
    .header-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 0;
        position: relative;
        z-index: 20;
    }}
    .logo-section {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }}
    .logo-shield {{
        width: 45px;
        height: 45px;
        background: linear-gradient(135deg, #4338ca, #7c3aed, {PRIMARY_COLOR});
        clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
    }}
    .logo-shield::before {{
        content: '⚔';
        font-size: 18px;
        position: absolute;
    }}
    .logo-text {{
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: 1px;
        color: {main_text_color};
    }}
    .nav-links-container {{
        display: flex;
        align-items: center;
        gap: 2.5rem;
    }}
    .stars-text {{
        color: {STARS_COLOR};
        font-size: 0.9rem;
        letter-spacing: 2px;
        white-space: nowrap;
    }}
    .main-content {{
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 2rem;
        position: relative;
        z-index: 10;
        max-width: 1000px;
        margin: 0 auto;
        min-height: calc(100vh - 120px);
    }}
    .hero-title {{
        font-family: 'Inter', sans-serif;
        font-size: clamp(2.8rem, 7vw, 5.5rem);
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 3rem;
        letter-spacing: -0.02em;
        color: {main_text_color};
    }}
    .hero-title .highlight {{
        background: linear-gradient(135deg, {HIGHLIGHT_GRADIENT_START} 0%, {HIGHLIGHT_GRADIENT_MIDDLE} 50%, {HIGHLIGHT_GRADIENT_END} 100%);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 4s ease-in-out infinite;
    }}
    .glow-orb {{
        position: absolute;
        border-radius: 50%;
        filter: blur(40px);
        opacity: 0.3;
        pointer-events: none;
        animation: pulse 4s ease-in-out infinite;
    }}
    .glow-1 {{ top: 20%; left: 10%; width: 200px; height: 200px; background: radial-gradient(circle, {HIGHLIGHT_GRADIENT_START} 0%, transparent 70%); animation-delay: -1s; }}
    .glow-2 {{ bottom: 30%; right: 15%; width: 150px; height: 150px; background: radial-gradient(circle, {HIGHLIGHT_GRADIENT_MIDDLE} 0%, transparent 70%); animation-delay: -2s; }}
    .glow-3 {{ top: 60%; left: 20%; width: 180px; height: 180px; background: radial-gradient(circle, {HIGHLIGHT_GRADIENT_END} 0%, transparent 70%); animation-delay: -3s; }}
    .crystal-placeholder {{
        position: absolute;
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.8), rgba(59, 130, 246, 0.9), rgba(236, 72, 153, 0.8));
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        opacity: 0.6;
        pointer-events: none;
        z-index: 5;
        animation: float 8s ease-in-out infinite;
    }}
    .crystal-diamond-p {{ top: 15%; left: 15%; width: 120px; height: 120px; clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%); animation-delay: -2s; }}
    .crystal-octagon-p {{ top: 25%; right: 12%; width: 100px; height: 100px; clip-path: polygon(30% 0%, 70% 0%, 100% 30%, 100% 70%, 70% 100%, 30% 100%, 0% 70%, 0% 30%); animation-delay: -4s; }}
    .crystal-sphere-p {{ bottom: 25%; right: 15%; width: 80px; height: 80px; border-radius: 50%; animation-delay: -1s; }}
    .crystal-ring-p {{ bottom: 20%; left: 8%; width: 150px; height: 150px; border-radius: 50%; border: 20px solid transparent; background-clip: padding-box; background: conic-gradient(from 0deg, rgba(139, 92, 246, 0.6) 0deg, rgba(59, 130, 246, 0.8) 120deg, rgba(236, 72, 153, 0.6) 240deg, rgba(139, 92, 246, 0.6) 360deg); animation-delay: -3s; }}
    .crystal-cube-p {{ top: 45%; left: 3%; width: 70px; height: 70px; animation-delay: -5s; }}
    .crystal-triangle-p {{ top: 55%; right: 8%; width: 90px; height: 90px; clip-path: polygon(50% 0%, 0% 100%, 100% 100%); animation-delay: -6s; }}
    
    /* Chatbot specific styling */
    .chat-header {{
        text-align: center; padding: 2rem 0;
        background: linear-gradient(90deg, {PRIMARY_COLOR}, {SECONDARY_COLOR}, {ACCENT_COLOR}, {PRIMARY_COLOR});
        background-size: 300% 300%; -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; font-family: 'Orbitron', monospace; font-weight: 900; font-size: 3.5rem;
        text-shadow: 0 0 30px rgba(5, 150, 105, 0.5); margin-bottom: 1rem;
    }}
    .sub-header {{
        text-align: center; color: {main_text_color}; opacity: 0.7;
        font-family: 'Rajdhani', sans-serif; font-size: 1.2rem;
        margin-bottom: 2rem; padding: 0 2rem;
    }}
    .stSidebar {{ background: {surface_color_chat}; backdrop-filter: blur(10px); }}
    .sidebar-header {{
        color: {sidebar_header_color}; font-family: 'Orbitron', monospace;
        font-weight: 700; font-size: 1.3rem; text-align: center;
        margin-bottom: 1rem; text-shadow: 0 0 10px rgba(5, 150, 105, 0.5);
    }}
    .feature-list {{
        background: {surface_color_chat}; border-left: 3px solid {primary_color_chat};
        padding: 1rem; margin: 1rem 0; border-radius: 0 8px 8px 0;
        font-family: 'Rajdhani', sans-serif; color: {sidebar_text_color};
    }}
    .status-success {{
        color: {primary_color_chat}; background: {surface_color_chat}; padding: 0.5rem 1rem;
        border-radius: 6px; border-left: 4px solid {primary_color_chat};
        font-family: 'Rajdhani', sans-serif; font-weight: 600;
    }}
    .status-error {{
        color: {accent_color_chat}; background: {surface_color_chat}; padding: 0.5rem 1rem;
        border-radius: 6px; border-left: 4px solid {accent_color_chat};
        font-family: 'Rajdhani', sans-serif; font-weight: 600;
    }}
    
    /* Fix selectbox label colors for light mode */
    .stSelectbox label {{
        color: {toggle_label_color} !important;
    }}
    .stSidebar .stMarkdown {{
        color: {sidebar_text_color} !important;
    }}
    .stSidebar p {{
        color: {sidebar_text_color} !important;
    }}
    .stSidebar div[data-testid="stMarkdownContainer"] {{
        color: {sidebar_text_color} !important;
    }}
    
    .stChatMessage {{ background: transparent !important; border: none !important; padding: 0 !important; margin: 1rem 0 !important; }}
    .stChatMessage > div:first-child {{ display: none !important; }}
    .user-message {{
        background: {surface_color_chat};
        border-left: 4px solid {primary_color_chat}; border-radius: 0 12px 12px 0;
        padding: 1rem 1.5rem; margin: 1rem 0; position: relative;
        font-family: 'Rajdhani', monospace; box-shadow: 0 0 20px {border_color_chat};
        backdrop-filter: blur(5px); color: {text_color_chat};
    }}
    .user-message::before {{
        content: ">"; position: absolute; left: -2px; top: 50%; transform: translateY(-50%);
        background: {primary_color_chat}; color: {button_text_color_chat}; width: 20px;
        height: 20px; border-radius: 50%; display: flex; align-items: center;
        justify-content: center; font-weight: bold; font-size: 12px;
        box-shadow: 0 0 10px rgba(5, 150, 105, 0.5);
    }}
    .user-message::after {{
        content: ""; position: absolute; right: -1px; top: 0; bottom: 0;
        width: 2px; background: linear-gradient(180deg, {primary_color_chat}, transparent);
    }}
    .ai-message {{
        background: {surface_color_chat};
        border: 1px solid {border_color_chat}; border-radius: 15px;
        padding: 1.5rem; margin: 1rem 0; position: relative;
        backdrop-filter: blur(10px); box-shadow: 0 8px 32px {border_color_chat};
        overflow: hidden; color: {text_color_chat};
    }}
    .ai-message::before {{
        content: ""; position: absolute; top: -2px; left: -2px; right: -2px; bottom: -2px;
        background: linear-gradient(45deg, {secondary_color_chat}, {accent_color_chat}, {primary_color_chat}, {secondary_color_chat});
        background-size: 300% 300%; border-radius: 15px; z-index: -1; opacity: 0.5;
    }}
    .ai-message::after {{
        content: "◇ KRYPTONIC AI"; position: absolute; top: -8px; left: 20px;
        background: linear-gradient(90deg, {secondary_color_chat}, {accent_color_chat});
        color: {button_text_color_chat}; padding: 2px 8px; font-size: 10px;
        font-weight: bold; border-radius: 4px; font-family: 'Orbitron', monospace;
        letter-spacing: 1px;
    }}
    .ai-message .data-stream {{
        position: absolute; right: 10px; top: 10px; width: 8px; height: 8px;
        background: {primary_color_chat}; border-radius: 50%; box-shadow: 0 0 10px {primary_color_chat};
    }}
    .ai-message .data-stream::before {{
        content: ""; position: absolute; right: 15px; top: 0; width: 6px; height: 6px;
        background: {secondary_color_chat}; border-radius: 50%; box-shadow: 0 0 8px {secondary_color_chat};
    }}
    .ai-message .data-stream::after {{
        content: ""; position: absolute; right: 25px; top: 1px; width: 4px; height: 4px;
        background: {accent_color_chat}; border-radius: 50%; box-shadow: 0 0 6px {accent_color_chat};
    }}
    .user-message .terminal-cursor {{
        display: inline-block; width: 2px; height: 1.2em;
        background: {primary_color_chat}; margin-left: 2px;
    }}
    .glow-text {{ text-shadow: 0 0 10px currentColor; }}
    .ai-message:hover {{ transform: translateY(-2px); box-shadow: 0 12px 40px {border_color_chat}; transition: all 0.3s ease; }}
    
    /* Control Button Wrapper Styling */
    .control-button-wrapper {{
        border-radius: 12px;
        padding: 2px;
        margin-bottom: 0.5rem;
    }}
    
    /* ON State Styling */
    .control-button-on .stButton > button {{
        background: {"linear-gradient(45deg, #00ff88, #00d4ff)" if is_dark_mode else "linear-gradient(45deg, #059669, #0284c7)"} !important;
        color: {"#000000" if is_dark_mode else "#ffffff"} !important;
        border: 2px solid {"#00ff88" if is_dark_mode else "#059669"} !important;
        border-radius: 12px !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        text-align: center !important;
        white-space: pre-line !important;
        box-shadow: 0 4px 12px {"rgba(0, 255, 136, 0.4)" if is_dark_mode else "rgba(5, 150, 105, 0.4)"} !important;
        transition: all 0.3s ease !important;
        opacity: 1 !important;
    }}
    
    .control-button-on .stButton > button:hover {{
        background: {"linear-gradient(45deg, #ff0080, #00ff88)" if is_dark_mode else "linear-gradient(45deg, #dc2626, #059669)"} !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px {"rgba(0, 255, 136, 0.6)" if is_dark_mode else "rgba(5, 150, 105, 0.6)"} !important;
    }}
    
    /* OFF State Styling */
    .control-button-off .stButton > button {{
        background: {"rgba(255, 255, 255, 0.15)" if is_dark_mode else "rgba(0, 0, 0, 0.15)"} !important;
        color: {"rgba(255, 255, 255, 0.7)" if is_dark_mode else "rgba(0, 0, 0, 0.7)"} !important;
        border: 2px solid {"rgba(255, 255, 255, 0.3)" if is_dark_mode else "rgba(0, 0, 0, 0.3)"} !important;
        border-radius: 12px !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        text-align: center !important;
        white-space: pre-line !important;
        box-shadow: 0 2px 6px {"rgba(0, 0, 0, 0.2)" if is_dark_mode else "rgba(0, 0, 0, 0.1)"} !important;
        transition: all 0.3s ease !important;
        opacity: 0.8 !important;
    }}
    
    .control-button-off .stButton > button:hover {{
        background: {"rgba(255, 255, 255, 0.25)" if is_dark_mode else "rgba(0, 0, 0, 0.25)"} !important;
        border-color: {"rgba(255, 255, 255, 0.5)" if is_dark_mode else "rgba(0, 0, 0, 0.5)"} !important;
        transform: translateY(-1px) !important;
        opacity: 0.9 !important;
    }}
    
    /* Chat History Styling */
    .history-item {{
        background: {surface_color_chat}; 
        border: 1px solid {border_color_chat};
        border-radius: 8px; 
        padding: 0.75rem; 
        margin-bottom: 0.5rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }}
    .history-item:hover {{
        border-color: {primary_color_chat};
        transform: translateY(-1px);
        box-shadow: 0 4px 12px {border_color_chat};
    }}
    .history-item-title {{
        color: {primary_color_chat}; 
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600; 
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }}
    .history-item-date {{
        color: {"rgba(255, 255, 255, 0.6)" if is_dark_mode else "rgba(0, 0, 0, 0.6)"};
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.75rem;
    }}
    .history-search {{
        background: {surface_color_chat} !important;
        border: 1px solid {border_color_chat} !important;
        color: {text_color_chat} !important;
        border-radius: 8px !important;
    }}
    .history-controls {{
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }}
    .stDataFrame {{ background: {surface_color_chat}; border-radius: 10px; border: 1px solid {border_color_chat}; }}
    .stButton > button {{
        background: linear-gradient(45deg, {primary_color_chat}, {secondary_color_chat});
        color: {button_text_color_chat}; border: none; border-radius: 8px;
        font-family: 'Rajdhani', sans-serif; font-weight: 600; font-size: 1rem;
        padding: 0.7rem 1.5rem; transition: all 0.3s ease;
        box-shadow: 0 4px 15px {border_color_chat}; text-transform: uppercase; letter-spacing: 1px;
    }}
    .stButton > button:hover {{
        background: linear-gradient(45deg, {accent_color_chat}, {primary_color_chat});
        transform: translateY(-2px); box-shadow: 6px 20px rgba(255, 0, 128, 0.4);
    }}


    
    /* Responsive design */
    @media (max-width: 768px) {{
        .main .block-container {{ padding-left: 1.5rem; padding-right: 1.5rem; }}
        .header-container {{ flex-direction: column; gap: 1rem; }}
        .nav-links-container {{ gap: 1.5rem; }}
        .hero-title {{ font-size: 3rem; margin-bottom: 2rem; }}
        .crystal-placeholder, .glow-orb {{ transform: scale(0.7); }}
    }}
    @media (max-width: 480px) {{
        .main .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
        .logo-text {{ font-size: 1.2rem; }}
        .nav-links-container {{ flex-wrap: wrap; justify-content: center; gap: 1rem; }}
        .hero-title {{ font-size: 2.2rem; }}
        .crystal-placeholder, .glow-orb {{ display: none; }}
        .modal-title {{ font-size: 1.5rem; }}
        .modal-text {{ font-size: 0.9rem; }}
    }}
    
    {animation_css}
    </style>
    """

def welcome_page():
    # --- Ambient Glow Effects and Crystal Shapes ---
    st.markdown(f'<div class="glow-orb glow-1"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glow-orb glow-2"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="glow-orb glow-3"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="crystal-placeholder crystal-diamond-p"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="crystal-placeholder crystal-octagon-p"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="crystal-placeholder crystal-sphere-p"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="crystal-placeholder crystal-ring-p"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="crystal-placeholder crystal-cube-p"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="crystal-placeholder crystal-triangle-p"></div>', unsafe_allow_html=True)

    # --- Header Section ---
    header_cols = st.columns([1, 2, 1])

    with header_cols[0]:
        st.markdown(
            """
            <div class="logo-section">
                <div class="logo-shield"></div>
                <div class="logo-text">CRYPTO KNIGHT</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with header_cols[1]:
        nav_cols = st.columns([1, 1, 1])
        with nav_cols[0]:
            if st.button("LOG IN", key="nav_login"):
                st.session_state.page = "login"
        with nav_cols[1]:
            st.markdown(f'<span class="stars-text">★★★★★</span>', unsafe_allow_html=True)
        with nav_cols[2]:
            if st.button("ABOUT", key="nav_about"):
                st.session_state.page = "about"

    with header_cols[2]:
        if st.button("Sign up →", key="signup_button"):
            st.session_state.page = "signup"

    # --- Main Content (Hero Section) ---
    st.markdown(
        f"""
        <div class="main-content">
            <h1 class="hero-title">
                Welcome to <span class="highlight">Crypto Knight</span>,<br>
                Your crypto guide, where<br>
                <span class="highlight">futures collide</span>
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Start a chat button ---
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button("Start a chat", key="get_started_btn"):
        st.session_state.page = "chatbot"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)



def chatbot_page():
    # --- Custom Header with crypto styling ---
    st.markdown("""
    <div style='text-align: center;'>
    <h1 class="chat-header">
        🚀 Kryptonic AI 🤖
    </h1>
    <div class="sub-header">
        ⚡ Your Crypto Buddy That Actually Gets It ⚡<br>
        <span style="color: #059669;">🔮 Smart AI • Live Prices • Easy Explanations 🔮</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar with quick actions and controls ---
    with st.sidebar:
        st.markdown('<div class="sidebar-header">⚙ CONTROLS ⚙</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            # Create wrapper div for styling
            dark_mode_class = "control-button-on" if st.session_state.dark_mode else "control-button-off"
            st.markdown(f'<div class="control-button-wrapper {dark_mode_class}">', unsafe_allow_html=True)
            dark_mode_text = f"🌙 Dark Mode\n{'ON' if st.session_state.dark_mode else 'OFF'}"
            if st.button(dark_mode_text, key="dark_toggle", use_container_width=True):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            # Create wrapper div for styling
            animations_class = "control-button-on" if st.session_state.animations_enabled else "control-button-off"
            st.markdown(f'<div class="control-button-wrapper {animations_class}">', unsafe_allow_html=True)
            animations_text = f"✨ Animations\n{'ON' if st.session_state.animations_enabled else 'OFF'}"
            if st.button(animations_text, key="anim_toggle", use_container_width=True):
                st.session_state.animations_enabled = not st.session_state.animations_enabled
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Add accessibility controls section
        st.markdown("---")
        st.markdown('<div class="sidebar-header">🛡️ ACCESSIBILITY 🛡️</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            reduce_motion_class = "control-button-on" if st.session_state.reduce_motion else "control-button-off"
            st.markdown(f'<div class="control-button-wrapper {reduce_motion_class}">', unsafe_allow_html=True)
            reduce_motion_text = f"🛡️ Reduce Motion\n{'ON' if st.session_state.reduce_motion else 'OFF'}"
            if st.button(reduce_motion_text, key="reduce_motion_toggle", use_container_width=True):
                st.session_state.reduce_motion = not st.session_state.reduce_motion
                # If reduce motion is turned on, also disable animations
                if st.session_state.reduce_motion:
                    st.session_state.animations_enabled = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            warning_class = "control-button-off"
            st.markdown(f'<div class="control-button-wrapper {warning_class}">', unsafe_allow_html=True)
            if st.button("⚠️ Show Warning\nAgain", key="show_warning", use_container_width=True):
                st.session_state.photosensitivity_warning_shown = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="sidebar-header">🌐 LANGUAGE 🌐</div>', unsafe_allow_html=True)
        
        # Use the expanded LANGUAGE_OPTIONS dictionary
        current_lang_name = next((name for name, code in LANGUAGE_OPTIONS.items() if code == st.session_state.selected_lang_code), 'English')
        default_index = list(LANGUAGE_OPTIONS.keys()).index(current_lang_name)
        selected_lang_name = st.selectbox(
            "Select Response Language",
            list(LANGUAGE_OPTIONS.keys()),
            index=default_index
        )
        
        # Update session state with the new language code
        st.session_state.selected_lang_code = LANGUAGE_OPTIONS[selected_lang_name]

        st.markdown("---")
        
        # Gamification Panel
        st.markdown('<div class="sidebar-header">🎮 PROGRESS 🎮</div>', unsafe_allow_html=True)
        
        # User stats
        user_stats = st.session_state.gamification.load_user_stats()
        xp_progress = (user_stats["total_xp"] % 100) / 100
        
        # Level display with animations
        level_emoji = "🌟" if user_stats['level'] < 5 else "💎" if user_stats['level'] < 10 else "👑" if user_stats['level'] < 15 else "🚀"
        xp_to_next = 100 - (user_stats['total_xp'] % 100)
        
        st.markdown(f"""
        <div style="text-align: center; padding: 1.2rem; background: {SURFACE_COLOR}; border-radius: 12px; border: 2px solid {BORDER_COLOR}; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0, 255, 136, 0.2);">
            <h2 style="margin: 0; color: {PRIMARY_COLOR}; font-family: 'Orbitron', monospace;">{level_emoji} Level {user_stats['level']} {level_emoji}</h2>
            <p style="margin: 0.5rem 0; color: {SECONDARY_COLOR}; font-weight: bold;">{user_stats['total_xp']} XP Total</p>
            <div style="background: rgba(255,255,255,0.15); border-radius: 12px; height: 12px; margin: 0.8rem 0; border: 1px solid rgba(255,255,255,0.3);">
                <div style="background: linear-gradient(90deg, {PRIMARY_COLOR}, {SECONDARY_COLOR}); height: 100%; border-radius: 12px; width: {xp_progress*100}%; transition: all 0.8s ease; box-shadow: 0 0 8px rgba(0, 255, 136, 0.5);"></div>
            </div>
            <p style="margin: 0; font-size: 0.9rem; color: {ACCENT_COLOR}; font-weight: 600;">⚡ {xp_to_next} XP to Level {user_stats['level'] + 1}!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced streak info
        streak_color = ACCENT_COLOR if user_stats['current_streak'] > 0 else "rgba(255, 255, 255, 0.5)"
        streak_message = f"🔥 {user_stats['current_streak']} day streak!" if user_stats['current_streak'] > 0 else "💤 No streak yet - come back tomorrow!"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 0.8rem; background: {SURFACE_COLOR}; border-radius: 10px; margin-bottom: 1rem; border: 1px solid {BORDER_COLOR};">
            <p style="margin: 0; font-size: 1rem; color: {streak_color}; font-weight: 600;">{streak_message}</p>
            {"<p style='margin: 0.3rem 0 0 0; font-size: 0.8rem; color: rgba(255,255,255,0.7);'>🏆 Best: " + str(user_stats['longest_streak']) + " days</p>" if user_stats['longest_streak'] > user_stats['current_streak'] else ""}
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced quick stats with progress indicators
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📬 Messages", user_stats['messages_sent'], delta=f"Goal: 50" if user_stats['messages_sent'] < 50 else "Goal reached!", delta_color="normal")
            st.metric("💎 Coins Learned", len(user_stats.get('coins_learned', [])), delta=f"Goal: 20" if len(user_stats.get('coins_learned', [])) < 20 else "Expert level!", delta_color="normal")
        with col2:
            st.metric("🏆 Challenges", user_stats.get('total_challenges_completed', 0), delta=f"Goal: 30" if user_stats.get('total_challenges_completed', 0) < 30 else "Master!", delta_color="normal")
            st.metric("👑 Best Streak", user_stats['longest_streak'], delta=f"Goal: 7 days" if user_stats['longest_streak'] < 7 else "Streak legend!", delta_color="normal")
        
        # Achievements toggle
        if st.button("🏆 View Achievements", key="show_achievements"):
            # Close other modals first
            st.session_state.show_daily_challenge = False
            st.session_state.challenge_submitted = False
            st.session_state.show_gamification = not st.session_state.show_gamification
            st.rerun()
        
        # Daily Challenge
        today = date.today().isoformat()
        if not user_stats.get('daily_challenge_completed') or user_stats.get('last_challenge_date') != today:
            if st.button("🧠 Daily Challenge", key="daily_challenge"):
                # Close other modals first
                st.session_state.show_gamification = False
                st.session_state.show_daily_challenge = True
                st.session_state.challenge_submitted = False
                st.rerun()
        else:
            st.success("✅ Today's challenge completed!")
        
        st.markdown("---")
        
        # Interactive Charts Panel
        st.markdown('<div class="sidebar-header">📊 LIVE CHARTS 📊</div>', unsafe_allow_html=True)
        
        # Initialize chart session state
        if 'show_charts' not in st.session_state:
            st.session_state.show_charts = False
        if 'selected_chart_coin' not in st.session_state:
            st.session_state.selected_chart_coin = 'bitcoin'
        if 'chart_timeframe' not in st.session_state:
            st.session_state.chart_timeframe = 7
        
        # Chart controls
        popular_coins = {
            'Bitcoin': 'bitcoin',
            'Ethereum': 'ethereum', 
            'BNB': 'binancecoin',
            'Cardano': 'cardano',
            'Solana': 'solana',
            'Polkadot': 'polkadot',
            'Dogecoin': 'dogecoin',
            'Chainlink': 'chainlink'
        }
        
        # Rate limit status indicator
        if hasattr(st.session_state, 'last_api_call'):
            time_since_last = time.time() - st.session_state.last_api_call
            if time_since_last < 5:
                remaining = 5 - time_since_last
                st.info(f"⏱️ API cooldown: {remaining:.1f}s remaining")
        
        # Quick chart buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📈 BTC Chart", use_container_width=True):
                st.session_state.selected_chart_coin = 'bitcoin'
                st.session_state.show_charts = True
                st.rerun()
                
        with col2:
            if st.button("📊 ETH Chart", use_container_width=True):
                st.session_state.selected_chart_coin = 'ethereum'
                st.session_state.show_charts = True
                st.rerun()
        
        # Main charts button
        if st.button("🎯 Open Charts Panel", key="show_charts_panel", use_container_width=True):
            st.session_state.show_charts = not st.session_state.show_charts
            st.rerun()
        
        if st.session_state.show_charts:
            st.markdown("**Chart Settings:**")
            
            # Coin selection
            coin_name = st.selectbox(
                "Select Cryptocurrency:",
                list(popular_coins.keys()),
                index=list(popular_coins.values()).index(st.session_state.selected_chart_coin) if st.session_state.selected_chart_coin in popular_coins.values() else 0,
                key="chart_coin_select"
            )
            st.session_state.selected_chart_coin = popular_coins[coin_name]
            
            # Timeframe selection
            timeframe_options = {
                '1 Day': 1,
                '1 Week': 7,
                '1 Month': 30,
                '3 Months': 90,
                '1 Year': 365
            }
            
            timeframe_name = st.selectbox(
                "Select Timeframe:",
                list(timeframe_options.keys()),
                index=1,  # Default to 1 week
                key="chart_timeframe_select"
            )
            st.session_state.chart_timeframe = timeframe_options[timeframe_name]
            
            # Cache status
            if 'price_cache' in st.session_state and st.session_state.selected_chart_coin in st.session_state.price_cache:
                cached_data, cache_time = st.session_state.price_cache[st.session_state.selected_chart_coin]
                cache_age = time.time() - cache_time
                if cache_age < 30:
                    st.success(f"📋 Using cached data ({30-cache_age:.0f}s fresh)")
            
            # Refresh chart button
            if st.button("🔄 Refresh Chart", use_container_width=True):
                # Clear cache to force fresh data
                if 'price_cache' in st.session_state:
                    st.session_state.price_cache.clear()
                if 'historical_cache' in st.session_state:
                    st.session_state.historical_cache.clear()
                st.success("🔄 Cache cleared! Loading fresh data...")
                st.rerun()
            
            # Diagnostic tools for troubleshooting
            with st.expander("🔧 Chart Troubleshooting"):
                st.session_state.chatbot.show_api_diagnostic()

        st.markdown("---")
        st.markdown('<div class="sidebar-header">⚡ QUICK STUFF ⚡</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔥 TRENDING"):
                response = st.session_state.chatbot.handle_trending_query()
                translated_response = translate_text(response, st.session_state.selected_lang_code)
                st.session_state.messages.append({"role": "assistant", "content": translated_response})
                # Track gamification
                st.session_state.gamification.record_market_view()
                st.session_state.gamification.add_xp(5, "Checked trending coins!")
                st.rerun()
            
            if st.button("💰 BTC PRICE"):
                response = st.session_state.chatbot.handle_price_query("bitcoin")
                translated_response = translate_text(response, st.session_state.selected_lang_code)
                st.session_state.messages.append({"role": "assistant", "content": translated_response})
                # Track gamification
                st.session_state.gamification.record_price_check("bitcoin")
                st.session_state.gamification.add_xp(5, "Checked Bitcoin price!")
                st.rerun()
        
        with col2:
            if st.button("📊 TOP COINS"):
                # Handle market overview properly
                message_text = "📊 Top 10 Biggest Cryptos Right Now:\n\n*These are ranked by how much they're worth in total! 💎*"
                translated_message = translate_text(message_text, st.session_state.selected_lang_code)
                
                df = st.session_state.chatbot.get_market_overview_df()
                
                if not df.empty:
                    st.session_state.messages.append({"role": "assistant", "content": translated_message, "type": "text"})
                    st.session_state.messages.append({"role": "assistant", "content": df, "type": "dataframe"})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": translate_text("Oops! Can't load the market data right now 📊 Give it another shot!", st.session_state.selected_lang_code), "type": "text"})
                
                # Track gamification
                st.session_state.gamification.record_market_view()
                st.session_state.gamification.add_xp(10, "Viewed top cryptocurrencies!")
                
                st.rerun()
            
            if st.button("🔄 CLEAR CHAT"):
                st.session_state.messages = []
                st.rerun()
        
        st.markdown("---")
        
        # --- Chat History Section ---
        st.markdown('<div class="sidebar-header">💾 CHAT HISTORY 💾</div>', unsafe_allow_html=True)
        
        # History controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Chat", key="save_chat_btn", use_container_width=True, disabled=len(st.session_state.messages) == 0):
                if st.session_state.messages:
                    saved_path = st.session_state.history_manager.save_chat_session(st.session_state.messages)
                    if saved_path:
                        st.success("💾 Chat saved!")
                        time.sleep(1)
                        st.rerun()
        with col2:
            history_toggle_text = f"📜 {'Hide' if st.session_state.show_history else 'Show'}"
            if st.button(history_toggle_text, key="toggle_history", use_container_width=True):
                st.session_state.show_history = not st.session_state.show_history
                st.rerun()
        
        # Show history if toggled on
        if st.session_state.show_history:
            # Search bar
            search_query = st.text_input("🔍 Search history", placeholder="Search chats...", key="history_search")
            
            # Load and display chat sessions
            sessions = st.session_state.history_manager.load_chat_sessions()
            
            if search_query:
                sessions = st.session_state.history_manager.search_chat_history(search_query, sessions)
            
            if sessions:
                st.markdown(f"<div style='font-size: 0.8rem; color: rgba(255,255,255,0.6); margin-bottom: 0.5rem;'>Found {len(sessions)} chat(s)</div>", unsafe_allow_html=True)
                
                # Display sessions with pagination
                max_sessions_per_page = 5
                for i, session in enumerate(sessions[:max_sessions_per_page]):
                    session_name = session.get('session_name', 'Untitled Chat')
                    created_date = session.get('created_at', '')
                    if created_date:
                        try:
                            date_obj = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
                            display_date = date_obj.strftime('%m/%d %H:%M')
                        except:
                            display_date = created_date[:16]
                    else:
                        display_date = 'Unknown'
                    
                    # Create a container for each history item
                    history_container = st.container()
                    with history_container:
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            if st.button(f"📄 {session_name}", key=f"load_{i}", use_container_width=True):
                                st.session_state.messages = session['messages']
                                st.session_state.show_history = False
                                st.success(f"✅ Loaded: {session_name}")
                                time.sleep(1)
                                st.rerun()
                        
                        with col2:
                            if st.button("📤", key=f"export_{i}", help="Export chat"):
                                exported_text = st.session_state.history_manager.export_chat_as_text(session)
                                st.download_button(
                                    label="⬇️ Download",
                                    data=exported_text,
                                    file_name=f"chat_{session.get('timestamp', 'unknown')}.txt",
                                    mime="text/plain",
                                    key=f"download_{i}"
                                )
                        
                        with col3:
                            if st.button("🗑️", key=f"delete_{i}", help="Delete chat"):
                                if st.session_state.history_manager.delete_chat_session(session['filepath']):
                                    st.success("🗑️ Deleted!")
                                    time.sleep(1)
                                    st.rerun()
                        
                        st.markdown(f'<div class="history-item-date">{display_date}</div>', unsafe_allow_html=True)
                
                if len(sessions) > max_sessions_per_page:
                    st.markdown(f"<div style='text-align: center; font-size: 0.8rem; color: rgba(255,255,255,0.5);'>...and {len(sessions) - max_sessions_per_page} more</div>", unsafe_allow_html=True)
            else:
                st.markdown("📭 No chat history yet. Start chatting to create some!")
                st.markdown("""
                <div style='font-size: 0.75rem; color: rgba(255,255,255,0.5); margin-top: 0.5rem;'>
                💡 <strong>Tip:</strong> Chats auto-save every 10 messages!<br>
                💾 Manual save anytime with the Save button<br>
                📜 Click Show to view & load previous chats
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
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
    
    # Display chat messages with custom styling
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <strong style="color: #00ff88; font-family: 'Orbitron', monospace;">USER_INPUT:</strong> {message["content"]}
                <div class="terminal-cursor"></div>
            </div>
            """, unsafe_allow_html=True)
        else: # role == "assistant"
            if message.get("type") == "dataframe":
                st.dataframe(message["content"], use_container_width=True)
            else:
                st.markdown(f"""
                <div class="ai-message">
                    <div class="data-stream"></div>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Add text-to-speech for AI responses if TTS is enabled
                if st.session_state.get('tts_enabled', False):
                    st.session_state.chatbot.create_text_to_speech_component(message["content"])
    
    # Text-to-Speech Toggle (Simple feature)
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tts_enabled = st.toggle("🔊 Enable AI Voice (Text-to-Speech)", 
                               value=st.session_state.get('tts_enabled', False),
                               help="When enabled, you can click 'Listen' buttons to hear AI responses")
        st.session_state.tts_enabled = tts_enabled
        
        if tts_enabled:
            st.success("🔊 AI can now speak! Look for 'Listen' buttons below responses.")
        else:
            st.info("🔇 AI voice disabled - responses will be text-only.")
    
    # Quick Actions Section (Simplified Alternative to Voice Features)
    st.markdown("---")
    
    with st.expander("⚡ Quick Actions", expanded=False):
        st.markdown("### 🚀 One-Click Crypto Commands")
        st.info("💡 **No complex setup needed** - Just click any button below to instantly execute that command!")
        
        # Main quick action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**💰 Prices**")
            if st.button("Bitcoin Price", key="quick_btc_price"):
                st.session_state.voice_command = "What's the Bitcoin price?"
                st.success("✅ Bitcoin price query queued!")
                
            if st.button("Ethereum Price", key="quick_eth_price"):
                st.session_state.voice_command = "What's the Ethereum price?"
                st.success("✅ Ethereum price query queued!")
                
            if st.button("Top 10 Prices", key="quick_top_prices"):
                st.session_state.voice_command = "Show market overview"
                st.success("✅ Market overview queued!")
        
        with col2:
            st.markdown("**📊 Charts & Analysis**")
            if st.button("Bitcoin Chart", key="quick_btc_chart"):
                st.session_state.show_charts = True
                st.session_state.selected_chart_coin = 'bitcoin'
                st.session_state.chart_timeframe = 7
                st.success("✅ Bitcoin chart loaded!")
                st.rerun()
                
            if st.button("Ethereum Chart", key="quick_eth_chart"):
                st.session_state.show_charts = True
                st.session_state.selected_chart_coin = 'ethereum'
                st.session_state.chart_timeframe = 7
                st.success("✅ Ethereum chart loaded!")
                st.rerun()
                
            if st.button("Compare BTC vs ETH", key="quick_btc_eth_compare"):
                st.session_state.show_charts = True
                st.session_state.selected_chart_coin = 'bitcoin'  # Start with bitcoin
                st.session_state.chart_timeframe = 7
                # Set a flag to show the comparison tab
                st.session_state.show_comparison_tab = True
                st.success("✅ BTC vs ETH comparison loaded!")
                st.rerun()
        
        with col3:
            st.markdown("**🔥 Trending & Info**")
            if st.button("What's Trending?", key="quick_trending_now"):
                st.session_state.voice_command = "What's trending in crypto?"
                st.success("✅ Trending cryptos queued!")
                
            if st.button("Crypto Basics", key="quick_crypto_help"):
                st.session_state.voice_command = "What is cryptocurrency and how does it work?"
                st.success("✅ Crypto education queued!")
                
            if st.button("Market News", key="quick_market_news"):
                st.session_state.voice_command = "What's happening in the crypto market today?"
                st.success("✅ Market news queued!")
        
        st.markdown("---")
        st.info("⚡ **Quick Actions**: Instant crypto commands! Toggle 'AI Voice' above to hear responses spoken aloud.")
        
        # Expandable command list
        with st.expander("📚 More Commands You Can Type"):
            st.markdown("""
            **Price Queries:**
            - "What's the [coin name] price?"
            - "Show me prices for Bitcoin, Ethereum, Solana"
            - "How much is Dogecoin worth?"
            
            **Charts & Visualization:**
            - "Show Bitcoin chart for the last 30 days"
            - "Display Ethereum price history"
            - "Visualize Solana performance"
            
            **Comparisons:**
            - "Compare Bitcoin vs Ethereum"
            - "Which is better: Solana or Cardano?"
            - "Show differences between BTC and ETH"
            
            **General Information:**
            - "What is blockchain technology?"
            - "Explain DeFi to me"
            - "How does cryptocurrency mining work?"
            - "What are NFTs?"
            
            **Market Analysis:**
            - "What's the crypto market looking like?"
            - "Show me trending cryptocurrencies"
            - "Which coins are performing well?"
            """)
        
    st.markdown("---")
    
    # Handle queued voice commands
    if hasattr(st.session_state, 'voice_command') and st.session_state.voice_command:
        voice_cmd = st.session_state.voice_command
        st.session_state.voice_command = None  # Clear the command
        
        # Add the voice command to messages and process it
        st.session_state.messages.append({"role": "user", "content": voice_cmd})
        st.success(f"🎤 Voice command executed: {voice_cmd}")
        
        # Track gamification for voice commands
        st.session_state.gamification.record_message(voice_cmd)
        st.session_state.gamification.add_xp(2, "Used a quick action!")
        
        # Display the user's message
        st.markdown(f"""
        <div class="user-message">
            <strong style="color: #00ff88; font-family: 'Orbitron', monospace;">USER_INPUT:</strong> {voice_cmd}
            <div class="terminal-cursor"></div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🧠 Thinking about your question..."):
            # Process the voice command
            response = st.session_state.chatbot.process_query(voice_cmd, st.session_state.selected_lang_code, st.session_state.messages)
            
            # Handle special case for market overview
            if response == "market_overview_requested":
                message_text = "📊 Top 10 Biggest Cryptos Right Now:\n\n*These are ranked by how much they're worth in total! 💎*"
                translated_message = translate_text(message_text, st.session_state.selected_lang_code)
                
                df = st.session_state.chatbot.get_market_overview_df()
                
                if not df.empty:
                    st.session_state.messages.append({"role": "assistant", "content": translated_message, "type": "text"})
                    st.session_state.messages.append({"role": "assistant", "content": df, "type": "dataframe"})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": translate_text("Oops! Can't load the market data right now 📊 Give it another shot!", st.session_state.selected_lang_code), "type": "text"})
            else:
                # Regular response
                translated_response = translate_text(response, st.session_state.selected_lang_code)
                st.session_state.messages.append({"role": "assistant", "content": translated_response, "type": "text"})
        
        st.rerun()  # Rerun to show the complete conversation
    
    # Chat input with enhanced styling
    if prompt := st.chat_input("🤑Ask me anything about crypto..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Track gamification for user messages
        st.session_state.gamification.record_message(prompt)
        st.session_state.gamification.add_xp(2, "Sent a message!")
        
        # Display the user's message
        st.markdown(f"""
        <div class="user-message">
            <strong style="color: #00ff88; font-family: 'Orbitron', monospace;">USER_INPUT:</strong> {prompt}
            <div class="terminal-cursor"></div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🧠 Thinking about your question..."):
            # Pass the conversation history to process_query
            # Pass the language code from session state to the processing function
            response = st.session_state.chatbot.process_query(prompt, st.session_state.selected_lang_code, st.session_state.messages)
            
            # Handle special case for market overview
            if response == "market_overview_requested":
                message_text = "📊 Top 10 Biggest Cryptos Right Now:\n\n*These are ranked by how much they're worth in total! 💎*"
                translated_message = translate_text(message_text, st.session_state.selected_lang_code)
                
                df = st.session_state.chatbot.get_market_overview_df()
                
                if not df.empty:
                    st.session_state.messages.append({"role": "assistant", "content": translated_message, "type": "text"})
                    st.session_state.messages.append({"role": "assistant", "content": df, "type": "dataframe"})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": translate_text("Oops! Can't load the market data right now 📊 Give it another shot!", st.session_state.selected_lang_code), "type": "text"})
            else:
                # Regular response
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Auto-save chat if it reaches certain length (every 10 messages)
        if len(st.session_state.messages) >= 10 and len(st.session_state.messages) % 10 == 0:
            saved_path = st.session_state.history_manager.save_chat_session(
                st.session_state.messages, 
                f"Auto-saved chat ({len(st.session_state.messages)} messages)"
            )
            if saved_path:
                st.toast("💾 Chat auto-saved!", icon="💾")
        
        st.rerun()

    # Achievements Panel (using Streamlit's native expander)
    if st.session_state.get('show_gamification', False):
        st.markdown("---")
        
        # Create a prominent achievements section
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {BG_GRADIENT_COLOR_1}, {BG_GRADIENT_COLOR_3}); 
                    padding: 2rem; border-radius: 15px; border: 2px solid {PRIMARY_COLOR};
                    box-shadow: 0 0 20px rgba(0, 255, 136, 0.3); margin: 2rem 0;">
            <h1 style="color: {PRIMARY_COLOR}; text-align: center; margin-bottom: 1.5rem;">🏆 Your Achievements</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Close button prominently at the top
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("❌ Close Achievements Panel", key="close_achievements", type="primary", use_container_width=True):
                st.session_state.show_gamification = False
                st.rerun()
        
        st.markdown("---")
        
        achievements_def = st.session_state.gamification.get_achievements_definition()
        achievements = st.session_state.gamification.load_achievements()
        
        unlocked_count = len(achievements["unlocked"])
        total_count = len(achievements_def)
        
        # Progress overview
        progress_percent = unlocked_count / total_count if total_count > 0 else 0
        st.subheader(f"📊 Progress Overview")
        st.progress(progress_percent)
        st.write(f"**{unlocked_count}/{total_count} Achievements Unlocked** ({progress_percent:.1%})")
        
        st.markdown("---")
        
        # Create two columns for unlocked and locked achievements
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌟 Unlocked Achievements")
            if achievements["unlocked"]:
                for achievement_key in achievements["unlocked"]:
                    if achievement_key in achievements_def:
                        ach = achievements_def[achievement_key]
                        st.success(f"{ach['icon']} **{ach['name']}**\n\n{ach['description']}\n\n*Reward: +{ach['xp_reward']} XP*")
            else:
                st.info("🎯 No achievements unlocked yet!\n\nKeep chatting and exploring to earn your first achievement!")
        
        with col2:
            st.subheader("🔒 Locked Achievements")
            locked_achievements = [key for key in achievements_def.keys() if key not in achievements["unlocked"]]
            if locked_achievements:
                for achievement_key in locked_achievements[:5]:  # Show first 5 locked achievements
                    ach = achievements_def[achievement_key]
                    st.info(f"🔒 **{ach['name']}**\n\n{ach['description']}\n\n*Reward: +{ach['xp_reward']} XP*")
                
                if len(locked_achievements) > 5:
                    with st.expander(f"View {len(locked_achievements) - 5} more locked achievements"):
                        for achievement_key in locked_achievements[5:]:
                            ach = achievements_def[achievement_key]
                            st.info(f"🔒 **{ach['name']}** - {ach['description']} (+{ach['xp_reward']} XP)")
            else:
                st.success("🎉 All achievements unlocked!\n\nYou're a Kryptonic master!")
        
        st.markdown("---")
        
        # Another close button at the bottom
        if st.button("🚪 Close and Continue Chatting", key="close_achievements_bottom", type="primary", use_container_width=True):
            st.session_state.show_gamification = False
            st.rerun()
    
    # Daily Challenge Section (using native Streamlit)
    if st.session_state.get('show_daily_challenge', False):
        challenge = st.session_state.gamification.get_daily_challenge()
        
        st.markdown("---")
        
        # Create a prominent challenge section
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {BG_GRADIENT_COLOR_2}, {BG_GRADIENT_COLOR_4}); 
                    padding: 2rem; border-radius: 15px; border: 2px solid {SECONDARY_COLOR};
                    box-shadow: 0 0 20px rgba(0, 212, 255, 0.3); margin: 2rem 0;">
            <h1 style="color: {SECONDARY_COLOR}; text-align: center; margin-bottom: 0.5rem;">🧠 Daily Challenge</h1>
            <p style="color: {TEXT_COLOR}; text-align: center; margin-bottom: 1rem; font-style: italic;">Test your crypto knowledge and earn XP!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Close button prominently at the top
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("❌ Close Challenge Panel", key="close_challenge_main", type="secondary", use_container_width=True):
                st.session_state.show_daily_challenge = False
                st.session_state.challenge_submitted = False
                st.session_state.challenge_result = None
                st.rerun()
        
        st.markdown("---")
        
        # Display the question in a nice format
        st.markdown(f"""
        <div style="background: {SURFACE_COLOR}; padding: 1.5rem; border-radius: 10px; border-left: 4px solid {SECONDARY_COLOR};">
            <h3 style="color: {SECONDARY_COLOR}; margin-bottom: 1rem;">📝 Today's Question:</h3>
            <p style="color: {TEXT_COLOR}; font-size: 1.1rem; font-weight: 500;">{challenge['question']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Handle answer submission with proper state management
        if not st.session_state.challenge_submitted:
            st.subheader("Choose your answer:")
            user_answer = st.radio(
                "Select one option:",
                challenge['options'], 
                key="challenge_answer",
                label_visibility="collapsed"
            )
            
            st.markdown("")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ Submit Answer", key="submit_challenge", type="primary", use_container_width=True):
                    selected_index = challenge['options'].index(user_answer)
                    is_correct = selected_index == challenge['correct']
                    st.session_state.challenge_submitted = True
                    st.session_state.challenge_result = is_correct
                    st.rerun()
            
            with col2:
                if st.button("⏭️ Skip Challenge", key="skip_challenge", type="secondary", use_container_width=True):
                    st.warning("⏭️ Challenge skipped. Come back tomorrow for a new challenge!")
                    time.sleep(2)
                    st.session_state.show_daily_challenge = False
                    st.session_state.challenge_submitted = False
                    st.rerun()
            
            with col3:
                if st.button("🚪 Exit", key="exit_challenge", type="secondary", use_container_width=True):
                    st.session_state.show_daily_challenge = False
                    st.session_state.challenge_submitted = False
                    st.rerun()
        
        else:
            # Show results
            is_correct = st.session_state.challenge_result
            
            st.markdown("### 📊 Results")
            
            if is_correct:
                st.success("🎉 **Correct Answer!** 🎉")
                st.balloons()
            else:
                st.error("❌ **Incorrect Answer**")
            
            # Show explanation in a nice format
            st.markdown(f"""
            <div style="background: {SURFACE_COLOR}; padding: 1.5rem; border-radius: 10px; border-left: 4px solid {PRIMARY_COLOR}; margin: 1rem 0;">
                <h4 style="color: {PRIMARY_COLOR}; margin-bottom: 1rem;">💡 Explanation:</h4>
                <p style="color: {TEXT_COLOR}; line-height: 1.6;">{challenge['explanation']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Complete the challenge and show XP reward
            completed = st.session_state.gamification.complete_daily_challenge(is_correct)
            if completed:
                xp_earned = challenge['xp_reward'] if is_correct else 10
                st.success(f"🏆 **Challenge Completed!** +{xp_earned} XP earned!")
                
                # Show motivational message
                if is_correct:
                    st.info("🌟 Great job! Your crypto knowledge is growing!")
                else:
                    st.info("📚 Every attempt makes you smarter! Keep learning!")
            
            st.markdown("---")
            
            # Continue button
            if st.button("🎯 Continue Learning & Close", key="continue_learning", type="primary", use_container_width=True):
                st.session_state.show_daily_challenge = False
                st.session_state.challenge_submitted = False
                st.session_state.challenge_result = None
                st.success("🚀 Keep exploring crypto with Kryptonic!")
                time.sleep(1)
                st.rerun()
    
    # Interactive Charts Panel
    if st.session_state.get('show_charts', False):
        st.markdown("---")
        
        # Create a prominent charts section
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {BG_GRADIENT_COLOR_1}, {BG_GRADIENT_COLOR_2}); 
                    padding: 2rem; border-radius: 15px; border: 2px solid {PRIMARY_COLOR};
                    box-shadow: 0 0 20px rgba(0, 255, 136, 0.3); margin: 2rem 0;">
            <h1 style="color: {PRIMARY_COLOR}; text-align: center; margin-bottom: 0.5rem;">📊 Interactive Crypto Charts</h1>
            <p style="color: {TEXT_COLOR}; text-align: center; margin-bottom: 1rem; font-style: italic;">Live price data with zoom, pan, and hover functionality</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Action buttons prominently at the top
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("🔄 REFRESH DATA", key="refresh_chart_data", use_container_width=True, type="primary"):
                # Clear all caches to force fresh data
                if 'price_cache' in st.session_state:
                    st.session_state.price_cache.clear()
                if 'historical_cache' in st.session_state:
                    st.session_state.historical_cache.clear()
                # Reset API call time to allow immediate requests
                st.session_state.last_api_call = 0
                st.success("🔄 All data refreshed! Loading fresh charts...")
                st.rerun()
        
        with col3:
            if st.button("❌ CLOSE & CONTINUE", key="close_charts_main", type="secondary", use_container_width=True):
                st.session_state.show_charts = False
                st.rerun()
        
        st.markdown("---")
        
        # Chart tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Single Coin", "📊 Compare Coins", "🎯 Market Overview"])
        
        with tab1:
            st.subheader(f"📈 {st.session_state.selected_chart_coin.replace('-', ' ').title()} Chart")
            
            # Display current price info
            price_data = st.session_state.chatbot.get_crypto_price(st.session_state.selected_chart_coin)
            if price_data and st.session_state.selected_chart_coin in price_data:
                coin_data = price_data[st.session_state.selected_chart_coin]
                current_price = coin_data['usd']
                change_24h = coin_data.get('usd_24h_change', 0)
                market_cap = coin_data.get('usd_market_cap', 0)
                volume_24h = coin_data.get('usd_24h_vol', 0)
                
                # Price metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("💰 Current Price", f"${current_price:,.2f}")
                with col2:
                    st.metric("📈 24h Change", f"{change_24h:+.2f}%", delta=f"{change_24h:+.2f}%")
                with col3:
                    st.metric("💎 Market Cap", f"${market_cap:,.0f}")
                with col4:
                    st.metric("📊 24h Volume", f"${volume_24h:,.0f}")
            
            st.markdown("")
            
            # Create and display the interactive chart
            with st.spinner(f"Loading {st.session_state.selected_chart_coin.title()} chart..."):
                chart = st.session_state.chatbot.create_interactive_chart(
                    st.session_state.selected_chart_coin, 
                    st.session_state.selected_chart_coin,
                    st.session_state.chart_timeframe
                )
                
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                    
                    # Chart features info
                    st.info("📌 **Chart Features:** Zoom with mouse wheel, pan by dragging, hover for details, double-click to reset zoom")
                    
                    # Track gamification
                    st.session_state.gamification.record_market_view()
                    st.session_state.gamification.add_xp(8, f"Viewed {st.session_state.selected_chart_coin.title()} chart!")
                else:
                    # Show fallback information when chart fails to load
                    st.session_state.chatbot.show_chart_fallback(
                        st.session_state.selected_chart_coin, 
                        st.session_state.selected_chart_coin
                    )
        
        with tab2:
            st.subheader("📊 Compare Multiple Cryptocurrencies")
            
            # Multi-select for comparison
            comparison_coins = {
                'Bitcoin': 'bitcoin',
                'Ethereum': 'ethereum', 
                'BNB': 'binancecoin',
                'Cardano': 'cardano',
                'Solana': 'solana',
                'Polkadot': 'polkadot',
                'Dogecoin': 'dogecoin',
                'Chainlink': 'chainlink'
            }
            
            # Handle quick action for BTC vs ETH comparison
            if hasattr(st.session_state, 'show_comparison_tab') and st.session_state.show_comparison_tab:
                default_selection = ['Bitcoin', 'Ethereum']
                st.session_state.show_comparison_tab = False  # Reset the flag
                st.info("🚀 **Quick Action**: Showing Bitcoin vs Ethereum comparison as requested!")
            else:
                default_selection = ['Bitcoin', 'Ethereum', 'BNB']
            
            selected_coins = st.multiselect(
                "Select cryptocurrencies to compare (2-5 recommended):",
                list(comparison_coins.keys()),
                default=default_selection,
                max_selections=5
            )
            
            if len(selected_coins) >= 2:
                coin_ids = [comparison_coins[coin] for coin in selected_coins]
                
                with st.spinner("Loading comparison chart..."):
                    comparison_chart = st.session_state.chatbot.create_comparison_chart(
                        coin_ids, selected_coins, st.session_state.chart_timeframe
                    )
                    
                    if comparison_chart:
                        st.plotly_chart(comparison_chart, use_container_width=True)
                        st.info("📊 **Comparison Chart:** Shows percentage change from starting point - perfect for comparing performance!")
                        
                        # Track gamification
                        st.session_state.gamification.record_market_view()
                        st.session_state.gamification.add_xp(15, f"Compared {len(selected_coins)} cryptocurrencies!")
                    # If comparison_chart is None, the create_comparison_chart method already showed detailed error messages
            else:
                st.warning("⚠️ Please select at least 2 cryptocurrencies to compare.")
        
        with tab3:
            st.subheader("🎯 Top 10 Market Overview")
            
            # Get market overview data
            market_df = st.session_state.chatbot.get_market_overview_df()
            
            if not market_df.empty:
                st.dataframe(
                    market_df,
                    use_container_width=True,
                    height=400
                )
                
                st.markdown("---")
                
                # Create a simple market cap chart
                fig = px.bar(
                    market_df.head(10),
                    x='Coin',
                    y='Market Cap',
                    title="Top 10 Cryptocurrencies by Market Cap",
                    template="plotly_dark"
                )
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color=TEXT_COLOR),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Track gamification
                st.session_state.gamification.record_market_view()
                st.session_state.gamification.add_xp(10, "Viewed market overview!")
            else:
                st.error("❌ Unable to load market data.")
        
        st.markdown("---")
        
        # Chart control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.toast("🔄 Refreshing data...", icon="🔄")
                # Clear caches to force fresh data
                if 'price_cache' in st.session_state:
                    st.session_state.price_cache.clear()
                if 'historical_cache' in st.session_state:
                    st.session_state.historical_cache.clear()
                st.rerun()
        
        with col2:
            if st.button("🚪 Close & Continue", key="close_charts_bottom", use_container_width=True):
                st.session_state.show_charts = False
                st.rerun()

    # Back to welcome page button
    st.markdown("<div style='text-align: center; margin-top: 2rem;'>", unsafe_allow_html=True)
    if st.button("Go back to Welcome Page", key="back_to_welcome"):
        st.session_state.page = "welcome"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def show_photosensitivity_warning():
    """Display photosensitivity warning dialog with integrated buttons."""
    st.markdown("""
    <div style="
        position: fixed; 
        top: 0; 
        left: 0; 
        width: 100%; 
        height: 100%; 
        background: rgba(0, 0, 0, 0.95); 
        z-index: 9999; 
        display: flex; 
        align-items: center; 
        justify-content: center;
        backdrop-filter: blur(10px);
    ">
        <div style="
            background: linear-gradient(135deg, #1a1a2e, #16213e); 
            padding: 2rem; 
            border-radius: 15px; 
            max-width: 500px; 
            margin: 1rem;
            border: 2px solid #ff6b6b;
            box-shadow: 0 0 30px rgba(255, 107, 107, 0.3);
        ">
            <h2 style="color: #ff6b6b; text-align: center; margin-bottom: 1rem;">⚠️ Photosensitivity Warning</h2>
            <p style="color: white; line-height: 1.6; margin-bottom: 1.5rem;">
                This application contains <strong>flashing lights, color transitions, and animated visual effects</strong> 
                that may potentially trigger seizures for individuals with photosensitive epilepsy or other light sensitivities.
            </p>
            <p style="color: white; line-height: 1.6; margin-bottom: 1.5rem;">
                <strong>If you are sensitive to flashing lights or have a history of seizures:</strong><br>
                • Consider using the "Reduce Motion" option<br>
                • Take regular breaks when using the app<br>
                • Consult with a medical professional if concerned
            </p>
            <div style="text-align: center;">
                <p style="color: #ffeb3b; font-size: 0.9rem; margin-bottom: 1.5rem;">
                    You can disable animations at any time in the sidebar settings.
                </p>
                <div style="display: flex; gap: 1rem; justify-content: center; margin-top: 1.5rem;">
                    <a href="?continue_with_effects=true" style="
                        background: linear-gradient(45deg, #00ff88, #00d4ff); 
                        color: #000000; 
                        padding: 0.75rem 1.5rem; 
                        border-radius: 12px; 
                        text-decoration: none; 
                        font-weight: 700; 
                        font-size: 0.9rem;
                        transition: all 0.3s ease;
                        border: 2px solid #00ff88;
                        display: inline-block;
                    " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 25px rgba(0, 255, 136, 0.4)'" onmouseout="this.style.transform=''; this.style.boxShadow=''">
                        ⚡ Continue with Effects
                    </a>
                    <a href="?reduce_motion=true" style="
                        background: linear-gradient(45deg, #ff6b6b, #ff8e53); 
                        color: white; 
                        padding: 0.75rem 1.5rem; 
                        border-radius: 12px; 
                        text-decoration: none; 
                        font-weight: 700; 
                        font-size: 0.9rem;
                        transition: all 0.3s ease;
                        border: 2px solid #ff6b6b;
                        display: inline-block;
                    " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 25px rgba(255, 107, 107, 0.4)'" onmouseout="this.style.transform=''; this.style.boxShadow=''">
                        🛡️ Reduce Motion
                    </a>
                </div>
                <div style="text-align: center; margin-top: 1rem;">
                    <p style="color: rgba(255, 255, 255, 0.6); font-size: 0.8rem;">
                        💡 If your browser/OS has "Reduce Motion" enabled, we recommend choosing "Reduce Motion"
                    </p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit application."""
    
    # --- Session State Initialization ---
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = True
    if 'animations_enabled' not in st.session_state:
        st.session_state.animations_enabled = True
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'selected_lang_code' not in st.session_state:
        st.session_state.selected_lang_code = 'en'
    if 'page' not in st.session_state:
        st.session_state.page = "welcome"
    if 'history_manager' not in st.session_state:
        st.session_state.history_manager = ChatHistoryManager()
    if 'show_history' not in st.session_state:
        st.session_state.show_history = False
    if 'photosensitivity_warning_shown' not in st.session_state:
        st.session_state.photosensitivity_warning_shown = False
    if 'reduce_motion' not in st.session_state:
        st.session_state.reduce_motion = False
    if 'gamification' not in st.session_state:
        st.session_state.gamification = GamificationManager()
        # Update daily activity on first load
        st.session_state.gamification.update_daily_activity()
    if 'show_gamification' not in st.session_state:
        st.session_state.show_gamification = False
    if 'show_daily_challenge' not in st.session_state:
        st.session_state.show_daily_challenge = False
    if 'challenge_submitted' not in st.session_state:
        st.session_state.challenge_submitted = False
    if 'challenge_result' not in st.session_state:
        st.session_state.challenge_result = None
    if 'show_charts' not in st.session_state:
        st.session_state.show_charts = False
    if 'selected_chart_coin' not in st.session_state:
        st.session_state.selected_chart_coin = 'bitcoin'
    if 'chart_timeframe' not in st.session_state:
        st.session_state.chart_timeframe = 7

    # Check for query parameters to handle warning choices
    try:
        # Try newer API first
        query_params = st.query_params
    except AttributeError:
        # Fallback to older API
        query_params = st.experimental_get_query_params()
    
    # Handle the choice from the warning dialog
    if 'continue_with_effects' in query_params:
        st.session_state.photosensitivity_warning_shown = True
        st.session_state.animations_enabled = True
        st.session_state.reduce_motion = False
        # Clear the query parameter
        try:
            st.query_params.clear()
        except AttributeError:
            st.experimental_set_query_params()
        st.rerun()
    elif 'reduce_motion' in query_params:
        st.session_state.photosensitivity_warning_shown = True
        st.session_state.animations_enabled = False
        st.session_state.reduce_motion = True
        # Clear the query parameter
        try:
            st.query_params.clear()
        except AttributeError:
            st.experimental_set_query_params()
        st.rerun()
    
    # Show photosensitivity warning on first visit
    if not st.session_state.photosensitivity_warning_shown:
        show_photosensitivity_warning()
        return

    # Apply theme CSS with motion settings
    st.markdown(get_theme_css(st.session_state.dark_mode, st.session_state.animations_enabled, st.session_state.reduce_motion), unsafe_allow_html=True)



    # Initialize chatbot instance if needed and on chatbot page
    if st.session_state.page == "chatbot" and st.session_state.chatbot is None:
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

    # Page navigation logic
    if st.session_state.page == "welcome":
        welcome_page()
    elif st.session_state.page == "chatbot":
        chatbot_page()
    elif st.session_state.page == "login":
        st.info("Login page coming soon!")
        if st.button("Back to Welcome"):
            st.session_state.page = "welcome"
            st.rerun()
    elif st.session_state.page == "about":
        st.info("About page coming soon!")
        if st.button("Back to Welcome"):
            st.session_state.page = "welcome"
            st.rerun()
    elif st.session_state.page == "signup":
        st.info("Sign up page coming soon!")
        if st.button("Back to Welcome"):
            st.session_state.page = "welcome"
            st.rerun()

if __name__ == "__main__":
    main()