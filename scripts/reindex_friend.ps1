param(
    [string]$LocalEsUrl = "http://localhost:9200",
    [string]$RemoteEsUrl = "http://100.99.130.69:9200",
    [string]$RemoteIndex = "flights",
    [string]$AliasName = "flights-remote",
    [int]$ReindexBatchSize = 5000,
    [int]$ReindexTimeoutSec = 1800,
    [string]$LogDir = "$PSScriptRoot"
)

# ── LOCK FILE ── cegah concurrent run
$LockFile = "$LogDir\reindex.lock"
if (Test-Path $LockFile) {
    $lockAge = (Get-Date) - (Get-Item $LockFile).LastWriteTime
    if ($lockAge.TotalMinutes -lt 30) {
        Write-Host "[LOCK] Previous reindex still running ($($lockAge.TotalMinutes.ToString('0.0'))m), skipping..."
        exit 0
    } else {
        Write-Host "[LOCK] Stale lock (>30m), removing..."
        Remove-Item $LockFile -Force
    }
}
Set-Content -Path $LockFile -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")

# Clean up lock on any exit
Register-EngineEvent -SourceIdentifier PowerShell.Exiting -SupportEvent -Action {
    if (Test-Path $LockFile) { Remove-Item $LockFile -Force }
} | Out-Null

$LogFile = "$LogDir\reindex.log"
$ErrorLogFile = "$LogDir\reindex_error.log"
$TimestampPattern = "yyyy-MM-dd HH:mm:ss"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $ts = Get-Date -Format $TimestampPattern
    $line = "[$ts] [$Level] $Message"
    $line | Out-File -FilePath $LogFile -Append -Encoding UTF8
    if ($Level -eq "ERROR") {
        $line | Out-File -FilePath $ErrorLogFile -Append -Encoding UTF8
        Write-Host $line -ForegroundColor Red
    } else {
        Write-Host $line
    }
}

function Invoke-EsApi {
    param([string]$Method, [string]$Uri, $Body, [int]$TimeoutSec = 60)
    $params = @{
        Method = $Method
        Uri = $Uri
        ContentType = "application/json"
        TimeoutSec = $TimeoutSec
    }
    if ($Body) {
        $bodyJson = $Body | ConvertTo-Json -Depth 10 -Compress
        $params.Body = $bodyJson
    }
    try {
        return Invoke-RestMethod @params
    } catch {
        $statusCode = $_.Exception.Response.StatusCode.value__
        $responseBody = ""
        try {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            $responseBody = $reader.ReadToEnd()
            $reader.Close()
        } catch {}
        Write-Log "HTTP $($_.Exception.Response.StatusCode) on $Method $Uri`n$responseBody" -Level "ERROR"
        throw
    }
}

Write-Log "============================================================"
Write-Log "Reindex Friend Flights - Started"
Write-Log "  Local ES: $LocalEsUrl"
Write-Log "  Remote ES: $RemoteEsUrl"
Write-Log "  Remote Index: $RemoteIndex"
Write-Log "  Alias: $AliasName"

# ---------------------------------------------------------------
# Step 1: Check remote ES reachability
# ---------------------------------------------------------------
try {
    $remoteHealth = Invoke-EsApi -Method Get -Uri "$RemoteEsUrl/"
    Write-Log "Remote ES connected: $($remoteHealth.name) ($($remoteHealth.version.number))"
} catch {
    Write-Log "Remote ES not reachable - skipping this cycle" -Level "WARN"
    exit 0
}

# ---------------------------------------------------------------
# Step 2: Determine generation for rolling swap
# ---------------------------------------------------------------
$currentGen = 0
try {
    $aliasResult = Invoke-EsApi -Method Get -Uri "$LocalEsUrl/_alias/$AliasName"
    foreach ($idx in $aliasResult.PSObject.Properties.Name) {
        if ($idx -match "^$AliasName-(\d+)$") {
            $g = [int]$Matches[1]
            if ($g -gt $currentGen) { $currentGen = $g }
        }
    }
} catch {
    Write-Log "No existing alias found, starting fresh" -Level "INFO"
}

$newGen = $currentGen + 1
$newIndexName = "$AliasName-$newGen"
Write-Log "  Current gen: $currentGen -> New gen: $newGen (index: $newIndexName)"

# ---------------------------------------------------------------
# Step 3: Get remote mapping and create new local index
# ---------------------------------------------------------------
Write-Log "Fetching remote mapping..."
try {
    $remoteMapping = Invoke-EsApi -Method Get -Uri "$RemoteEsUrl/$RemoteIndex/_mapping"
    $mappingSource = $remoteMapping.PSObject.Properties.Value.mappings
} catch {
    Write-Log "Using default mapping from mapping.json" -Level "WARN"
    $mappingPath = Join-Path $PSScriptRoot "..\mapping.json"
    if (Test-Path $mappingPath) {
        $mappingJsonRaw = Get-Content $mappingPath -Raw -Encoding UTF8
        $mappingFull = $mappingJsonRaw | ConvertFrom-Json
        $mappingSource = $mappingFull.flights.mappings
    } else {
        Write-Log "No mapping.json found, using minimal mapping" -Level "WARN"
        $mappingSource = @{
            properties = @{
                timestamp     = @{ type = "float" }
                ingested_at   = @{ type = "date" }
                processed_at  = @{ type = "date" }
                latitude      = @{ type = "float" }
                longitude     = @{ type = "float" }
                velocity      = @{ type = "float" }
                geo_altitude  = @{ type = "float" }
                true_track    = @{ type = "float" }
                vertical_rate = @{ type = "float" }
            }
        }
    }
}

# Delete existing index if present (dari run sebelumnya yang gagal)
try {
    $checkIdx = Invoke-EsApi -Method Get -Uri "$LocalEsUrl/$newIndexName"
    Write-Log "Index $newIndexName already exists, deleting..." -Level "WARN"
    Invoke-EsApi -Method Delete -Uri "$LocalEsUrl/$newIndexName"
    Write-Log "Deleted old $newIndexName"
} catch {
    # Index tidak ada — normal, lanjut buat baru
}

$createBody = @{ mappings = $mappingSource }
try {
    Invoke-EsApi -Method Put -Uri "$LocalEsUrl/$newIndexName" -Body $createBody
    Write-Log "Created index: $newIndexName"
} catch {
    Write-Log "Failed to create index $newIndexName - aborting" -Level "ERROR"
    exit 1
}

# ---------------------------------------------------------------
# Step 4: Reindex from remote to new local index
# ---------------------------------------------------------------
Write-Log "Starting reindex from $RemoteEsUrl/$RemoteIndex to $newIndexName ..."

$reindexBody = @{
    source = @{
        remote = @{ host = $RemoteEsUrl }
        index = $RemoteIndex
        size = $ReindexBatchSize
        query = @{ match_all = @{} }
    }
    dest = @{ index = $newIndexName }
}

$reindexTask = $null
try {
    $reindexTask = Invoke-EsApi -Method Post -Uri "$LocalEsUrl/_reindex?wait_for_completion=true" -Body $reindexBody -TimeoutSec $ReindexTimeoutSec
} catch {
    # Try async if sync times out
    Write-Log "Sync reindex timed out, trying async..." -Level "WARN"
    try {
        $reindexTask = Invoke-EsApi -Method Post -Uri "$LocalEsUrl/_reindex?wait_for_completion=false" -Body $reindexBody -TimeoutSec 30
        $taskId = $reindexTask.task
        Write-Log "Reindex task: $taskId, waiting for completion..."

        do {
            Start-Sleep -Seconds 10
            $taskStatus = Invoke-EsApi -Method Get -Uri "$LocalEsUrl/_tasks/$taskId"
            $status = $taskStatus.task.status
            $created = if ($status.created) { $status.created } else { 0 }
            $total = if ($status.total) { $status.total } else { 0 }
            Write-Log "  Progress: $created / $total docs"
            $completed = $taskStatus.task.completed
        } while (-not $completed)
        $reindexTask = $taskStatus
    } catch {
        Write-Log "Async reindex also failed: $_" -Level "ERROR"
        Invoke-EsApi -Method Delete -Uri "$LocalEsUrl/$newIndexName" | Out-Null
        exit 1
    }
}

$totalDocs = 0
if ($reindexTask -and $reindexTask.task -and $reindexTask.task.status) {
    $totalDocs = $reindexTask.task.status.created
    if (-not $totalDocs) { $totalDocs = $reindexTask.task.status.total }
}
Write-Log "Reindex complete: ~$totalDocs documents"

# ---------------------------------------------------------------
# Step 5: Verify document count
# ---------------------------------------------------------------
try {
    $countResult = Invoke-EsApi -Method Get -Uri "$LocalEsUrl/$newIndexName/_count"
    $actualDocs = $countResult.count
    Write-Log "Verified: $actualDocs documents in $newIndexName"
} catch {
    Write-Log "Could not verify count" -Level "WARN"
    $actualDocs = 0
}

# ---------------------------------------------------------------
# Step 6: Swap alias (zero-downtime)
# ---------------------------------------------------------------
Write-Log "Swapping alias '$AliasName' ..."

$aliasActions = @()

# Remove alias from old index (if exists)
if ($currentGen -gt 0) {
    $oldIndexName = "$AliasName-$currentGen"
    try {
        $aliasActions += @{
            remove = @{
                index = $oldIndexName
                alias = $AliasName
            }
        }
    } catch {
        Write-Log "Old index $oldIndexName not found, skipping removal" -Level "WARN"
    }
}

# Add alias to new index
$aliasActions += @{
    add = @{
        index = $newIndexName
        alias = $AliasName
    }
}

$aliasBody = @{ actions = $aliasActions }
try {
    Invoke-EsApi -Method Post -Uri "$LocalEsUrl/_aliases" -Body $aliasBody
    Write-Log "Alias '$AliasName' -> '$newIndexName'"
} catch {
    Write-Log "Alias swap failed: $_" -Level "ERROR"
    exit 1
}

# ---------------------------------------------------------------
# Step 7: Delete old index
# ---------------------------------------------------------------
if ($currentGen -gt 0) {
    $oldIndexName = "$AliasName-$currentGen"
    try {
        Invoke-EsApi -Method Delete -Uri "$LocalEsUrl/$oldIndexName"
        Write-Log "Deleted old index: $oldIndexName"
    } catch {
        Write-Log "Could not delete $oldIndexName (may have been deleted already)" -Level "WARN"
    }
}

# ---------------------------------------------------------------
# Done
# ---------------------------------------------------------------
# Remove lock file
if (Test-Path $LockFile) { Remove-Item $LockFile -Force }

Write-Log "SUCCESS: Reindexed $actualDocs docs to $newIndexName (alias: $AliasName)"
Write-Log "============================================================"
Write-Log ""
