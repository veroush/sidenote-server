# Sidenote startup script
#
# Launches all three processes Sidenote needs, each in its own window:
#   1. uvicorn  (main FastAPI server: transcription + note matching)
#   2. tailscale serve  (HTTPS tunnel so the phone mic can be used)
#   3. whatsapp-service  (Go/whatsmeow service for sending WhatsApp messages)
#
# Run this from anywhere -- it uses absolute paths below.
# EDIT the two paths below if this project ever moves.

$sidenoteServerPath = "C:\Projects\sidenote-server"
$whatsappServicePath = "C:\Projects\sidenote-server\whatsapp-service"

Write-Host "Starting Sidenote (3 windows will open)..." -ForegroundColor Cyan

# 1. Main FastAPI server (uvicorn)
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$sidenoteServerPath'; uvicorn server:app --host 0.0.0.0 --port 8000"
)

# 2. Tailscale HTTPS tunnel
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "tailscale serve 8000"
)

# 3. WhatsApp service (Go)
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$whatsappServicePath'; go run main.go"
)

Write-Host "All three windows launched. Check each for errors before using the app." -ForegroundColor Green
Write-Host "The WhatsApp service window will need a QR scan only if whatsapp-session.db is missing or expired." -ForegroundColor Yellow