# Start ROVY Cloud Server (runs on PC)
# This handles:
# - REST API (port 8000) - for mobile app
# - WebSocket (port 8765) - for robot communication
# - AI processing: LLM, Vision, STT, TTS

Set-Location $PSScriptRoot

Write-Host "================================"
Write-Host "  ROVY CLOUD SERVER (PC)"
Write-Host "================================"
Write-Host ""
Write-Host "This runs on your PC with GPU"
Write-Host ""

# Check Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "ERROR: Python not found" -ForegroundColor Red
    exit 1
}

# Get IP
$IP = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notlike "*Loopback*" -and $_.PrefixOrigin -ne "WellKnown" } | Select-Object -First 1).IPAddress
if (-not $IP) {
    $IP = "localhost"
}

Write-Host "Starting Cloud Server..."
Write-Host ""
Write-Host "  REST API:    http://${IP}:8000"
Write-Host "  WebSocket:   ws://${IP}:8765"
Write-Host "  Tailscale:   http://100.121.110.125:8000"
Write-Host ""
Write-Host "Logs will appear below:"
Write-Host ""

# Load environment variables from registry (User scope)
$envVars = [System.Environment]::GetEnvironmentVariables([System.EnvironmentVariableTarget]::User)
foreach ($key in $envVars.Keys) {
    if (-not [System.Environment]::GetEnvironmentVariable($key, [System.EnvironmentVariableTarget]::Process)) {
        [System.Environment]::SetEnvironmentVariable($key, $envVars[$key], [System.EnvironmentVariableTarget]::Process)
    }
}

python main.py

