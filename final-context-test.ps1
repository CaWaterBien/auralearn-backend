Write-Host "Testing AuraBot with all context types" -ForegroundColor Yellow

$htmlCode = "<!DOCTYPE html><html><head><title>My Page</title></head><body><header><h1>Welcome</h1></header><!-- Need to add main content --></body></html>"

$instructions = "Create semantic HTML page with header, navigation, main content sections, image with alt text, and contact form"

$feedback = "Previous feedback: Missing main element, No navigation in header, Score: 45/100"

$body = @{
    question = "What should I focus on next to improve my score?"
    html_context = $htmlCode
    instructions_context = $instructions
    feedback_context = $feedback
    session_id = "test_context"
    user_id = $null
} | ConvertTo-Json -Depth 3

try {
    $response = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/aurabot/ask' -Method POST -ContentType 'application/json' -Body $body
    Write-Host "SUCCESS:" -ForegroundColor Green
    Write-Host $response.response -ForegroundColor White
    
    Write-Host "`nAnalysis:" -ForegroundColor Yellow
    if ($response.response -match "main") { Write-Host "Mentions main element" -ForegroundColor Green }
    if ($response.response -match "nav") { Write-Host "Mentions navigation" -ForegroundColor Green }
    if ($response.response -match "header") { Write-Host "References existing header" -ForegroundColor Green }
    if ($response.response -match "score") { Write-Host "References score" -ForegroundColor Green }
    
} catch {
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
}

Read-Host "`nPress Enter to continue"
