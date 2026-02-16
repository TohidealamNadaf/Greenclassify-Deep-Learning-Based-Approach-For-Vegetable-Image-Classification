
# Check and Enable Long Paths
$longPathWait = 0
try {
    $longPaths = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -ErrorAction SilentlyContinue
    if ($longPaths -eq $null -or $longPaths.LongPathsEnabled -ne 1) {
        Write-Host "Enabling Long Paths (Requires Admin)..."
        try {
            New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force -ErrorAction Stop
            Write-Host "Long Paths Enabled."
        } catch {
            Write-Warning "Failed to enable Long Paths. Please run this script as Administrator or manually enable Long Paths."
            Write-Warning "If this step is skipped, TensorFlow installation might fail due to path length limitations."
            $longPathWait = 1
        }
    } else {
        Write-Host "Long Paths are already enabled."
    }
} catch {
    Write-Warning "Could not check Long Paths status."
}

# Create Virtual Environment if not exists
if (-not (Test-Path "venv")) {
    Write-Host "Creating Virtual Environment 'venv'..."
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
} else {
    Write-Host "Virtual Environment 'venv' already exists."
}

# Install Requirements
if (Test-Path "venv/Scripts/activate.ps1") {
    Write-Host "Activating venv and installing requirements..."
    & ./venv/Scripts/activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install requirements."
        if ($longPathWait -eq 1) {
             Write-Warning "Requirement installation failed possibly due to Long Path limitations. Try enabling Long Paths manually."
        }
        exit 1
    }
    Write-Host "Setup Complete."
} else {
    Write-Error "Could not find activation script at ./venv/Scripts/activate.ps1"
    exit 1
}
