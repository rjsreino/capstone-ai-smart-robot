# Stop ROVY Cloud Server (runs on PC)
# This stops the Python process running main.py

Set-Location $PSScriptRoot

Write-Host "================================"
Write-Host "  STOPPING ROVY CLOUD SERVER"
Write-Host "================================"
Write-Host ""

# Find Python processes running main.py
$allPythonProcs = Get-Process python,pythonw -ErrorAction SilentlyContinue
$processesToStop = @()

foreach ($proc in $allPythonProcs) {
    try {
        $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.Id)" | Select-Object -ExpandProperty CommandLine)
        if ($cmdLine -and $cmdLine -like "*main.py*" -and $cmdLine -like "*cloud*") {
            $processesToStop += @{
                Id = $proc.Id
                CommandLine = $cmdLine
            }
        }
    }
    catch {
        # Skip if we can't get command line
    }
}

if ($processesToStop.Count -gt 0) {
    Write-Host "Found $($processesToStop.Count) process(es) running main.py" -ForegroundColor Yellow
    Write-Host ""
    
    foreach ($procInfo in $processesToStop) {
        try {
            Write-Host "Stopping process: PID $($procInfo.Id)" -ForegroundColor Yellow
            $shortCmd = if ($procInfo.CommandLine.Length -gt 80) { 
                $procInfo.CommandLine.Substring(0, 80) + "..." 
            } else { 
                $procInfo.CommandLine 
            }
            Write-Host "  Command: $shortCmd" -ForegroundColor Gray
            Stop-Process -Id $procInfo.Id -Force -ErrorAction Stop
            Write-Host "  [OK] Process stopped" -ForegroundColor Green
        }
        catch {
            Write-Host "  [ERROR] Error stopping process: $_" -ForegroundColor Red
        }
    }
    
    Write-Host ""
    Write-Host "Checking if ports 8000 and 8765 are still in use..." -ForegroundColor Yellow
    
    # Wait a moment for ports to be released
    Start-Sleep -Seconds 2
    
    # Check if ports are still in use
    $port8000 = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
    $port8765 = Get-NetTCPConnection -LocalPort 8765 -ErrorAction SilentlyContinue
    
    if ($port8000 -or $port8765) {
        Write-Host "  [WARNING] Ports may still be in use. Waiting a bit longer..." -ForegroundColor Yellow
        Start-Sleep -Seconds 3
        
        $port8000 = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
        $port8765 = Get-NetTCPConnection -LocalPort 8765 -ErrorAction SilentlyContinue
        
        if ($port8000 -or $port8765) {
            Write-Host "  [WARNING] Some ports may still be in use. You may need to manually check." -ForegroundColor Yellow
        } else {
            Write-Host "  [OK] All ports released" -ForegroundColor Green
        }
    } else {
        Write-Host "  [OK] All ports released" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "Cloud server stopped successfully!" -ForegroundColor Green
}
else {
    Write-Host "No Python process running main.py found." -ForegroundColor Yellow
    Write-Host ""
    
    # Check if ports are in use - try to stop by port
    $port8000 = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -First 1
    $port8765 = Get-NetTCPConnection -LocalPort 8765 -ErrorAction SilentlyContinue | Select-Object -First 1
    
    if ($port8000 -or $port8765) {
        Write-Host "[WARNING] Ports 8000 or 8765 are still in use. Attempting to stop by port..." -ForegroundColor Yellow
        Write-Host ""
        
        $stoppedByPort = $false
        if ($port8000) {
            try {
                $pid8000 = $port8000.OwningProcess
                Write-Host "Stopping process on port 8000: PID $pid8000" -ForegroundColor Yellow
                Stop-Process -Id $pid8000 -Force -ErrorAction Stop
                Write-Host "  [OK] Process stopped" -ForegroundColor Green
                $stoppedByPort = $true
            }
            catch {
                Write-Host "  [ERROR] Error: $_" -ForegroundColor Red
            }
        }
        
        if ($port8765) {
            try {
                $pid8765 = $port8765.OwningProcess
                Write-Host "Stopping process on port 8765: PID $pid8765" -ForegroundColor Yellow
                Stop-Process -Id $pid8765 -Force -ErrorAction Stop
                Write-Host "  [OK] Process stopped" -ForegroundColor Green
                $stoppedByPort = $true
            }
            catch {
                Write-Host "  [ERROR] Error: $_" -ForegroundColor Red
            }
        }
        
        if ($stoppedByPort) {
            Write-Host ""
            Write-Host "Process stopped by port!" -ForegroundColor Green
        } else {
            Write-Host ""
            Write-Host "[WARNING] Could not stop process. You may need to manually check:" -ForegroundColor Yellow
            Write-Host "  Get-NetTCPConnection -LocalPort 8000" -ForegroundColor Gray
            Write-Host "  Get-NetTCPConnection -LocalPort 8765" -ForegroundColor Gray
        }
    } else {
        Write-Host "Cloud server is not running." -ForegroundColor Green
    }
}

Write-Host ""

