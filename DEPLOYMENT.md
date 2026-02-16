# Manipur PowerGuard - Deployment Guide

## 🚀 Deployment Options

### Option 1: Local Development (Quickest)

Start both servers manually:

```bash
# Terminal 1 - Backend
cd backend
python app.py

# Terminal 2 - Frontend  
cd frontend
npm start
```

Access at: http://localhost:3000

---

### Option 2: Local Production (Using Scripts)

```bash
# Make script executable
chmod +x run-local.sh

# Install dependencies
./run-local.sh install

# Build and run production
./run-local.sh prod

# Stop servers
./run-local.sh stop
```

---

### Option 3: Docker Deployment (Recommended for Production)

First, install Docker:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose-v2

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (logout/login after)
sudo usermod -aG docker $USER
```

Then deploy:
```bash
chmod +x deploy.sh
./deploy.sh deploy
```

Access at: http://localhost

Stop:
```bash
./deploy.sh stop
```

---

### Option 4: Cloud Deployment

#### Render.com (Free Tier Available)

1. **Backend (Web Service)**
   - Connect GitHub repo
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`

2. **Frontend (Static Site)**
   - Root Directory: `frontend`
   - Build Command: `npm install && npm run build`
   - Publish Directory: `build`
   - Add env var: `REACT_APP_API_URL=https://your-backend.onrender.com/api`

#### Railway.app

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

#### Vercel (Frontend) + Railway (Backend)

**Frontend on Vercel:**
```bash
cd frontend
npm install -g vercel
vercel
```

**Backend on Railway:**
```bash
cd backend
railway init
railway up
```

---

### Option 5: VPS Deployment (DigitalOcean, AWS, etc.)

1. **SSH into your server**
```bash
ssh user@your-server-ip
```

2. **Install requirements**
```bash
# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Install Python
sudo apt install -y python3 python3-pip python3-venv

# Install Nginx
sudo apt install -y nginx
```

3. **Clone and setup**
```bash
git clone <your-repo-url> powerguard
cd powerguard

# Backend setup
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn

# Frontend build
cd ../frontend
npm install
REACT_APP_API_URL=/api npm run build
```

4. **Configure Nginx** (`/etc/nginx/sites-available/powerguard`)
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        root /path/to/powerguard/frontend/build;
        try_files $uri $uri/ /index.html;
    }

    # API Proxy
    location /api {
        proxy_pass http://127.0.0.1:5000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

5. **Create systemd service** (`/etc/systemd/system/powerguard.service`)
```ini
[Unit]
Description=PowerGuard Backend
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/powerguard/backend
ExecStart=/path/to/venv/bin/gunicorn --bind 127.0.0.1:5000 --workers 4 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

6. **Enable and start**
```bash
sudo ln -s /etc/nginx/sites-available/powerguard /etc/nginx/sites-enabled/
sudo systemctl restart nginx
sudo systemctl enable powerguard
sudo systemctl start powerguard
```

---

## 🔒 Production Checklist

- [ ] Set `FLASK_ENV=production`
- [ ] Configure CORS for your domain
- [ ] Enable HTTPS (use Certbot for free SSL)
- [ ] Set up logging
- [ ] Configure firewall (UFW)
- [ ] Set up monitoring (optional)

## 📊 Health Check

Backend API health: `curl http://localhost:5000/api/health`

Expected response:
```json
{"status": "healthy", "service": "manipur-powerguard-api"}
```
