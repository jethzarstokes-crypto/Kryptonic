from IPython.display import display, HTML

display(HTML("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crypto Knight - Your Crypto Guide</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Rajdhani', sans-serif;
            background: linear-gradient(135deg, #1e1a3e 0%, #2d1b69 25%, #3e2a7a 50%, #4a1e8a 75%, #2a1454 100%);
            min-height: 100vh;
            overflow-x: hidden;
            color: white;
            position: relative;
        }

        /* Animated background elements */
        .bg-shapes {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            z-index: 1;
        }

        .shape {
            position: absolute;
            opacity: 0.1;
            animation: float 8s ease-in-out infinite;
        }

        .shape-1 {
            width: 200px;
            height: 200px;
            background: linear-gradient(45deg, #00ff88, #00d4ff);
            clip-path: polygon(50% 0%, 0% 100%, 100% 100%);
            top: 10%;
            left: 5%;
            animation-delay: 0s;
        }

        .shape-2 {
            width: 150px;
            height: 150px;
            background: linear-gradient(45deg, #ff0080, #00ff88);
            border-radius: 50%;
            top: 15%;
            right: 10%;
            animation-delay: 2s;
        }

        .shape-3 {
            width: 180px;
            height: 180px;
            background: linear-gradient(45deg, #00d4ff, #ff0080);
            clip-path: polygon(25% 0%, 100% 0%, 75% 100%, 0% 100%);
            bottom: 20%;
            left: 15%;
            animation-delay: 4s;
        }

        .shape-4 {
            width: 120px;
            height: 120px;
            background: linear-gradient(45deg, #00ff88, #ff0080);
            transform: rotate(45deg);
            bottom: 15%;
            right: 20%;
            animation-delay: 1s;
        }

        .shape-5 {
            width: 100px;
            height: 100px;
            background: linear-gradient(45deg, #00d4ff, #00ff88);
            clip-path: polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%);
            top: 50%;
            left: 80%;
            animation-delay: 3s;
        }

        @keyframes float {
            0%, 100% { 
                transform: translateY(0px) rotate(0deg); 
                opacity: 0.1;
            }
            25% { 
                transform: translateY(-30px) rotate(90deg); 
                opacity: 0.2;
            }
            50% { 
                transform: translateY(-15px) rotate(180deg); 
                opacity: 0.15;
            }
            75% { 
                transform: translateY(-45px) rotate(270deg); 
                opacity: 0.25;
            }
        }

        /* Navigation */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 2rem 4rem;
            position: relative;
            z-index: 10;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .logo-icon {
            width: 50px;
            height: 50px;
            background: linear-gradient(45deg, #00ff88, #00d4ff);
            clip-path: polygon(50% 0%, 0% 100%, 100% 100%);
            position: relative;
        }

        .logo-icon::after {
            content: '⚔';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 20px;
            color: #1e1a3e;
        }

        .logo-text {
            font-family: 'Orbitron', monospace;
            font-size: 1.5rem;
            font-weight: 700;
            color: white;
        }

        .nav-links {
            display: flex;
            gap: 3rem;
            align-items: center;
        }

        .nav-link {
            color: rgba(255, 255, 255, 0.8);
            text-decoration: none;
            font-weight: 500;
            transition: all 0.3s ease;
            position: relative;
        }

        .nav-link:hover {
            color: #00ff88;
        }

        .nav-link::after {
            content: '';
            position: absolute;
            bottom: -5px;
            left: 0;
            width: 0;
            height: 2px;
            background: linear-gradient(45deg, #00ff88, #00d4ff);
            transition: width 0.3s ease;
        }

        .nav-link:hover::after {
            width: 100%;
        }

        .signup-btn {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            padding: 0.8rem 1.5rem;
            border-radius: 25px;
            text-decoration: none;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }

        .signup-btn:hover {
            background: rgba(0, 255, 136, 0.2);
            border-color: #00ff88;
            color: #00ff88;
            transform: translateY(-2px);
        }

        /* Main content */
        .main-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 80vh;
            text-align: center;
            padding: 2rem;
            position: relative;
            z-index: 10;
        }

        .hero-title {
            font-size: 4rem;
            font-weight: 700;
            margin-bottom: 1rem;
            line-height: 1.2;
            max-width: 800px;
        }

        .hero-subtitle {
            font-size: 1.5rem;
            color: #00ff88;
            margin-bottom: 3rem;
            opacity: 0.9;
            max-width: 600px;
        }

        .start-chat-btn {
            display: inline-flex;
            align-items: center;
            gap: 1rem;
            background: linear-gradient(45deg, #00ff88, #00d4ff);
            color: #1e1a3e;
            padding: 1.5rem 3rem;
            border-radius: 50px;
            font-size: 1.2rem;
            font-weight: 700;
            text-decoration: none;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 30px rgba(0, 255, 136, 0.3);
            text-transform: uppercase;
            letter-spacing: 1px;
            position: relative;
            overflow: hidden;
        }

        .start-chat-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.4), transparent);
            transition: left 0.6s;
        }

        .start-chat-btn:hover {
            transform: translateY(-5px) scale(1.05);
            box-shadow: 0 20px 40px rgba(0, 255, 136, 0.5);
            background: linear-gradient(45deg, #ff0080, #00ff88);
        }

        .start-chat-btn:hover::before {
            left: 100%;
        }

        .btn-icon {
            font-size: 1.5rem;
        }

        /* Features section */
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 2rem;
            margin-top: 6rem;
            max-width: 1200px;
            width: 100%;
            position: relative;
            z-index: 10;
        }

        .feature-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(0, 255, 136, 0.2);
            border-radius: 20px;
            padding: 2.5rem;
            backdrop-filter: blur(15px);
            transition: all 0.4s ease;
            position: relative;
            overflow: hidden;
        }

        .feature-card::before {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, #00ff88, #00d4ff, #ff0080, #00ff88);
            background-size: 400% 400%;
            border-radius: 20px;
            z-index: -1;
            opacity: 0;
            animation: gradient-animation 4s ease infinite;
            transition: opacity 0.4s ease;
        }

        .feature-card:hover {
            transform: translateY(-15px);
            box-shadow: 0 25px 50px rgba(0, 255, 136, 0.3);
        }

        .feature-card:hover::before {
            opacity: 0.8;
        }

        @keyframes gradient-animation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        .feature-icon {
            font-size: 3rem;
            margin-bottom: 1.5rem;
            display: block;
        }

        .feature-title {
            font-family: 'Orbitron', monospace;
            font-size: 1.3rem;
            font-weight: 700;
            color: #00ff88;
            margin-bottom: 1rem;
        }

        .feature-description {
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.8);
            line-height: 1.6;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .navbar {
                padding: 1rem 2rem;
                flex-direction: column;
                gap: 1rem;
            }

            .nav-links {
                gap: 1.5rem;
            }

            .hero-title {
                font-size: 2.5rem;
            }

            .hero-subtitle {
                font-size: 1.2rem;
            }

            .start-chat-btn {
                padding: 1.2rem 2rem;
                font-size: 1rem;
            }

            .features {
                grid-template-columns: 1fr;
                margin-top: 3rem;
            }

            .feature-card {
                padding: 2rem;
            }
        }

        /* Glitch effect for logo */
        .logo-text {
            position: relative;
        }

        .logo-text::before,
        .logo-text::after {
            content: 'CRYPTO KNIGHT';
            position: absolute;
            top: 0;
            left: 0;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .logo-text::before {
            color: #ff0080;
            transform: translateX(-2px);
            animation: glitch-1 0.3s infinite;
        }

        .logo-text::after {
            color: #00d4ff;
            transform: translateX(2px);
            animation: glitch-2 0.3s infinite;
        }

        .logo:hover .logo-text::before,
        .logo:hover .logo-text::after {
            opacity: 0.7;
        }

        @keyframes glitch-1 {
            0%, 100% { transform: translateX(-2px); }
            50% { transform: translateX(-4px); }
        }

        @keyframes glitch-2 {
            0%, 100% { transform: translateX(2px); }
            50% { transform: translateX(4px); }
        }
    </style>
</head>
<body>
    <!-- Animated background shapes -->
    <div class="bg-shapes">
        <div class="shape shape-1"></div>
        <div class="shape shape-2"></div>
        <div class="shape shape-3"></div>
        <div class="shape shape-4"></div>
        <div class="shape shape-5"></div>
    </div>

    <!-- Navigation -->
    <nav class="navbar">
        <div class="logo">
            <div class="logo-icon"></div>
            <span class="logo-text">CRYPTO KNIGHT</span>
        </div>
        <div class="nav-links">
            <a href="#" class="nav-link">LOG IN</a>
            <span class="nav-link">★★★★★</span>
            <a href="#" class="nav-link">ABOUT</a>
            <a href="#" class="signup-btn">Sign up →</a>
        </div>
    </nav>

    <!-- Main content -->
    <main class="main-content">
        <h1 class="hero-title">
            Welcome to Crypto Knight, Your crypto guide, where futures collide
        </h1>
        <p class="hero-subtitle">
            Your AI-powered companion for navigating the crypto universe with confidence and intelligence
        </p>
        
        <button class="start-chat-btn" onclick="startChat()">
            <span class="btn-icon">💬</span>
            Start a chat
            <span class="btn-icon">→</span>
        </button>

        <!-- Features section -->
        <div class="features">
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <h3 class="feature-title">AI-Powered Insights</h3>
                <p class="feature-description">
                    Get smart, personalized crypto advice powered by advanced AI that understands market trends and your needs.
                </p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <h3 class="feature-title">Real-Time Data</h3>
                <p class="feature-description">
                    Access live crypto prices, trending coins, and market analysis updated in real-time for informed decisions.
                </p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🛡</div>
                <h3 class="feature-title">Safe & Secure</h3>
                <p class="feature-description">
                    Crypto-focused conversations only, with honest risk assessments and educational guidance you can trust.
                </p>
            </div>
        </div>
    </main>

    <script>
        function startChat() {
            // Add a cool transition effect
            const btn = document.querySelector('.start-chat-btn');
            btn.style.transform = 'scale(0.95)';
            btn.style.background = 'linear-gradient(45deg, #ff0080, #00d4ff)';
            
            setTimeout(() => {
                // Here you would redirect to your Streamlit chatbot
                // For demo purposes, we'll show an alert
                alert('🚀 Launching Crypto Knight Chatbot!\\n\\nIn a real implementation, this would redirect to your Streamlit app at:\\nhttp://localhost:8501');
                
                // In production, you would use:
                // window.location.href = 'http://localhost:8501';
                // or wherever your Streamlit app is hosted
            }, 200);
            
            setTimeout(() => {
                btn.style.transform = '';
                btn.style.background = '';
            }, 400);
        }

        // Add some interactive effects
        document.addEventListener('mousemove', (e) => {
            const shapes = document.querySelectorAll('.shape');
            const x = e.clientX / window.innerWidth;
            const y = e.clientY / window.innerHeight;
            
            shapes.forEach((shape, index) => {
                const speed = (index + 1) * 0.5;
                const xOffset = (x - 0.5) * speed * 20;
                const yOffset = (y - 0.5) * speed * 20;
                
                shape.style.transform = `translate(${xOffset}px, ${yOffset}px)`;
            });
        });

        // Add scroll effect to features
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, observerOptions);

        // Initially hide feature cards for animation
        document.querySelectorAll('.feature-card').forEach(card => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(50px)';
            card.style.transition = 'all 0.6s ease';
            observer.observe(card);
        });
    </script>
</body>
</html>
"""))
