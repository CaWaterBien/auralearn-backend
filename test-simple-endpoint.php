<?php
// Simple test to see what's happening with our endpoint
require_once 'vendor/autoload.php';

echo "🔧 Testing AI Validation System Directly\n";
echo "========================================\n\n";

try {
    // Test the service directly
    $nebiusClient = new App\Services\NebiusClient();
    $aiValidationService = new App\Services\AiValidationService($nebiusClient);
    
    echo "✅ Services instantiated successfully\n";
    
    // Test basic validation
    $result = $aiValidationService->validateCodeWithAi(
        'Hello World', 
        ['Create a basic HTML page'], 
        'Test Activity',
        'Test description'
    );
    
    echo "✅ AI validation completed\n";
    echo "Score: " . $result['overall_score'] . "\n";
    echo "AI Powered: " . ($result['ai_powered'] ? 'Yes' : 'No') . "\n";
    
} catch (Exception $e) {
    echo "❌ Error: " . $e->getMessage() . "\n";
    echo "File: " . $e->getFile() . ":" . $e->getLine() . "\n";
    
    // Also show the trace to see where it failed
    echo "\nStack trace:\n";
    foreach ($e->getTrace() as $i => $trace) {
        if (isset($trace['file'])) {
            echo "#$i " . $trace['file'] . "(" . $trace['line'] . "): ";
            if (isset($trace['class'])) {
                echo $trace['class'] . $trace['type'];
            }
            echo $trace['function'] . "()\n";
        }
        if ($i >= 5) break; // Limit output
    }
}
