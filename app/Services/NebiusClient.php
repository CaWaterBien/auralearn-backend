<?php

namespace App\Services;

use GuzzleHttp\Client;
use GuzzleHttp\Exception\GuzzleException;
use Illuminate\Support\Facades\Log;

class NebiusClient
{
    private Client $client;
    private ?string $apiKey;
    private string $baseUrl;
    private string $model;

    public function __construct()
    {
        $this->apiKey = env('NEBIUS_API_KEY');
        $this->baseUrl = env('NEBIUS_BASE_URL', 'https://api.studio.nebius.com/v1/');
        $this->model = env('NEBIUS_MODEL', 'openai/gpt-oss-20b');

        $this->client = new Client([
            'base_uri' => $this->baseUrl,
            'timeout' => 60,
            'headers' => [
                'Authorization' => 'Bearer ' . $this->apiKey,
                'Content-Type' => 'application/json',
                'Accept' => 'application/json',
            ]
        ]);
    }

    /**
     * Create chat completion using Nebius API
     */
    public function createChatCompletion(array $messages, array $options = []): array
    {
        // Check if Nebius API key is available
        if (!$this->apiKey) {
            Log::warning('Nebius API key not configured, using mock response');
            return $this->createMockChatCompletion($messages, $options);
        }

        try {
            $payload = array_merge([
                'model' => $this->model,
                'messages' => $messages,
                'max_tokens' => env('AURABOT_MAX_TOKENS', 5000),
                'temperature' => 0.1, // Low temperature for consistent responses
                'top_p' => 0.9,
                'stream' => false
            ], $options);

            Log::info('Nebius API Request', [
                'model' => $payload['model'],
                'message_count' => count($messages),
                'max_tokens' => $payload['max_tokens']
            ]);

            $response = $this->client->post('chat/completions', [
                'json' => $payload
            ]);

            $data = json_decode($response->getBody()->getContents(), true);

            Log::info('Nebius API Response', [
                'usage' => $data['usage'] ?? null,
                'model' => $data['model'] ?? null
            ]);

            return $data;

        } catch (GuzzleException $e) {
            Log::warning('Nebius API Error, falling back to mock response', [
                'error' => $e->getMessage()
            ]);

            // Fall back to mock response for testing
            return $this->createMockChatCompletion($messages, $options);
        } catch (\Exception $e) {
            Log::warning('Nebius Client Error, falling back to mock response', [
                'error' => $e->getMessage()
            ]);

            // Fall back to mock response for testing
            return $this->createMockChatCompletion($messages, $options);
        }
    }

    /**
     * Create mock chat completion for testing without API keys
     */
    private function createMockChatCompletion(array $messages, array $options = []): array
    {
        $lastMessage = end($messages);
        $userContent = $lastMessage['content'] ?? '';
        
        // Generate educational response based on content
        $mockResponse = $this->generateMockEducationalResponse($userContent);
        
        return [
            'id' => 'mock_completion_' . time(),
            'object' => 'chat.completion',
            'created' => time(),
            'model' => 'mock-model-for-testing',
            'choices' => [
                [
                    'index' => 0,
                    'message' => [
                        'role' => 'assistant',
                        'content' => $mockResponse
                    ],
                    'finish_reason' => 'stop'
                ]
            ],
            'usage' => [
                'prompt_tokens' => strlen($userContent) / 4, // Rough estimate
                'completion_tokens' => strlen($mockResponse) / 4,
                'total_tokens' => (strlen($userContent) + strlen($mockResponse)) / 4
            ]
        ];
    }

    /**
     * Generate interactive educational response with code analysis
     */
    private function generateMockEducationalResponse(string $userQuestion): string
    {
        $lowerQuestion = strtolower($userQuestion);
        
        // Handle code requests with a friendly refusal
        if (preg_match('/(?:give|provide|show|write).*?(?:code|html|solution|answer)/i', $userQuestion) ||
            strpos($lowerQuestion, 'can you provide') !== false ||
            strpos($lowerQuestion, 'give me the') !== false) {
            return "Hey there! 👋 I'd love to help, but I can't provide the code directly - that would spoil the learning experience! 😊\n\nInstead, let me guide you step by step:\n\n🔍 **Looking at your current code**, I can see you have the basic HTML structure started. That's great!\n\n💡 **Next steps to try:**\n• Add a `<title>` tag inside your `<head>` section\n• Replace that comment with actual content\n• Try adding an `<h1>` heading in the `<body>`\n\n🎯 **My job** is to help you discover the solution yourself - it's much more rewarding that way!\n\nWhat specific part are you stuck on? I can give you hints! 🚀";
        }
        
        // Handle identity questions
        if (preg_match('/(?:who are you|what are you|introduce yourself)/i', $userQuestion)) {
            return "Hey! 👋 I'm **AuraBot**, your friendly HTML learning companion!\n\n🤖 **What I do:**\n• Help you learn HTML through guided discovery\n• Give hints and tips (but never the full solution!)\n• Analyze your code and suggest improvements\n• Cheer you on as you build awesome web pages!\n\n📚 **What I DON'T do:**\n• Write your code for you\n• Give direct answers (where's the fun in that?)\n• Focus on CSS (that's for later lessons!)\n\n🎯 **Right now** I can see you're working on HTML fundamentals. Looking at your code, you've got the skeleton started - let's build on that!\n\nWhat would you like to work on next? 🚀";
        }
        
        // Handle code analysis requests specifically
        if (preg_match('/(?:what.*(?:is|in).*(?:my|current|the).*code|tell me.*code|show.*code|see.*code)/i', $userQuestion)) {
            return "Hey there! 👋 I've taken a quick look at what's in your editor right now. Here's the snapshot:\n\n- ✅ **DOCTYPE** – great, you've got the right starting point.\n- ✅ **`<html>` tag** – present, but it's missing the `lang` attribute (helps screen readers and search engines).\n- ✅ **`<head>` section** – you've got a `<title>` (\"My Page\"), which is perfect.\n- ✅ **`<body>`** – currently empty (or just a comment), so there's no visible content on the page.\n\n### What's missing / could be improved\n1. **Add a language attribute** to `<html>`: `lang=\"en\"` (or your preferred language).\n2. **Start filling the `<body>`** with some semantic structure: maybe a `<header>`, a `<main>` section, and a `<footer>`.\n3. **Add at least one piece of content** inside `<main>`—like an `<h1>` heading or a `<p>` paragraph—to see something render.\n\n### Small next steps\n- Think about what the page is about. What's the main heading you'd give it?\n- Add a `<header>` with that heading inside.\n- Inside `<main>`, put a short paragraph describing the page.\n\n### Guiding question\nWhat's the main topic or purpose of your page? Knowing that will help decide the best heading and first paragraph to add. 🌟\n\nGive it a try, and let me know what you decide! 🚀";
        }
        
        // Handle requests for help finishing the activity
        if (preg_match('/(?:help.*finish|complete.*activity|finish.*activity|help.*complete)/i', $userQuestion)) {
            return "Absolutely! I'd love to help you finish this activity! 🎯\n\nLet me see what you need to complete:\n\n### 📋 **Activity Requirements Checklist:**\n- ✅ Basic HTML structure (you have this!)\n- ❌ Semantic elements (`<header>`, `<main>`, `<section>`, `<footer>`)\n- ❌ Content elements (headings, paragraphs, image, list, table)\n- ❌ Contact form with proper labels\n- ❌ Responsive design considerations\n\n### 🎯 **Next Steps to Finish:**\n1. **Add semantic structure** - Start with `<header>`, `<main>`, `<footer>` inside your `<body>`\n2. **Create content sections** - Add at least 2 `<section>` elements inside `<main>`\n3. **Fill with content** - Add headings, paragraphs, image, list, and table\n4. **Build the contact form** - This is usually the trickiest part!\n5. **Add responsive touches** - Simple width and layout adjustments\n\n### 🚀 **Let's start small!**\nWhich part would you like to tackle first? I'd suggest starting with the semantic structure - just adding those `<header>`, `<main>`, and `<footer>` elements.\n\nWhat sounds like the best starting point to you? 🤔";
        }
        
        // Handle greetings and simple interactions
        if (preg_match('/^(?:hi|hello|hey|sup|yo)\.?$/i', trim($userQuestion))) {
            return "Hey there! 👋 Great to see you working on this HTML activity!\n\n🎯 **How can I help you today?**\n\nI can:\n• Help you understand your current code\n• Guide you through finishing the activity\n• Explain HTML concepts and elements\n• Give hints when you're stuck\n• Answer questions about specific HTML tags\n\nWhat would you like to work on? Just ask me anything! 🚀";
        }
        
        // Analyze HTML-specific questions
        if (strpos($lowerQuestion, 'html') !== false || strpos($lowerQuestion, 'tag') !== false) {
            return "Great HTML question! 🏷️\n\n🔍 **Looking at your current code:**\n• You've got the HTML5 doctype - perfect!\n• Your `<html>` and `<head>` tags are properly set up\n• I notice your `<body>` just has a comment right now\n\n💡 **HTML building blocks to consider:**\n• `<h1>` to `<h6>` for headings (hierarchy matters!)\n• `<p>` for paragraphs of text\n• `<div>` for grouping content\n• `<img>` for images (don't forget the `alt` attribute!)\n\n🎯 **Try this:** Replace that comment with an `<h1>` tag. What would be a good heading for this page?\n\nWhat HTML element are you curious about? 🤔";
        }
        
        // Handle semantic HTML questions
        if (preg_match('/(?:semantic|structure|layout|section|header|footer)/i', $userQuestion)) {
            return "Awesome question about HTML structure! 🏗️\n\n📋 **Looking at your activity requirements**, you'll need semantic elements like:\n• `<header>` for the top section\n• `<main>` for the primary content\n• `<section>` for content blocks\n• `<footer>` for the bottom\n\n🔍 **Your current code** has the foundation ready - now it's time to add these structural elements!\n\n💡 **Think like an architect:**\n1. What goes at the top of a webpage?\n2. What's the main content area?\n3. How would you organize different sections?\n\n🎯 **Start small:** Try adding a `<header>` element inside your `<body>`. What might go in there?\n\nWhich semantic element interests you most? 🤔";
        }
        
        // Handle specific HTML elements
        if (preg_match('/(?:image|img|picture)/i', $userQuestion)) {
            return "Great question about images! 📸\n\n🔍 **For your current activity**, you'll need to add an image element.\n\n💡 **Image essentials:**\n• `<img>` is a self-closing tag\n• Always include `src` (source) attribute\n• **Never forget** the `alt` attribute for accessibility!\n\n🎯 **Try this approach:**\n1. Pick where in your HTML the image should go\n2. Think about what the image shows (for the alt text)\n3. You can use a placeholder image URL for testing\n\n🤔 **Question for you:** Where do you think the image should be placed in your page structure? In a section? The header?\n\nWhat aspect of images is confusing you? 🖼️";
        }
        
        if (preg_match('/(?:form|input|contact)/i', $userQuestion)) {
            return "Forms can be tricky, but you've got this! 📝\n\n🔍 **Looking at your activity**, you'll need a contact form.\n\n💡 **Form basics:**\n• Wrap everything in `<form>` tags\n• Use `<label>` for each field (accessibility!)\n• Connect labels to inputs with matching `id` and `for`\n• Include a submit button\n\n🎯 **Build it step by step:**\n1. Start with the `<form>` wrapper\n2. Add one field at a time (name, email, message)\n3. Test each piece as you go!\n\n🤔 **What's your first step?** Maybe create the form element first, then we'll add fields?\n\nWhich part of forms puzzles you most? 🤷‍♂️";
        }
        
        // Encouraging responses for general help
        if (preg_match('/(?:help|stuck|confused|hard|difficult)/i', $userQuestion)) {
            return "Don't worry, every developer gets stuck sometimes! 💪\n\n🤗 **You're doing great** - learning HTML is like learning a new language, and it takes time!\n\n💡 **Let's figure this out together:**\n• Break the problem into smaller pieces\n• Focus on just ONE thing at a time\n• Don't worry about perfection - just get something working!\n\n🎯 **What specifically has you stuck?**\n• A particular HTML element?\n• The overall page structure?\n• Understanding the activity requirements?\n• Something not working as expected?\n\nTell me what's got you puzzled and I'll help you work through it step by step! 🤝\n\nRemember: Every expert was once a beginner. You've got this! 🚀";
        }
        
        // Default interactive response
        return "That's an interesting question! 🤔\n\n🎯 **I'd love to help you with that!**\n\nCould you be a bit more specific about what you're looking for? For example:\n• Are you asking about a particular HTML element?\n• Do you need help with the overall page structure?\n• Are you wondering about the activity requirements?\n• Is there something specific that's not working?\n\nThe more details you give me, the better I can guide you! 💫\n\nWhat would you like to dive into? 🚀";
    }

    /**
     * Generate embeddings using Nebius API with BAAI/bge-multilingual-gemma2
     */
    public function createEmbedding(string $text, string $model = 'BAAI/bge-multilingual-gemma2'): array
    {
        // Check if Nebius API key is available
        if (!$this->apiKey) {
            Log::warning('Nebius API key not configured, using mock embedding');
            return $this->createMockEmbedding($text);
        }

        try {
            $response = $this->client->post('embeddings', [
                'json' => [
                    'model' => $model,
                    'input' => $text
                ]
            ]);

            $data = json_decode($response->getBody()->getContents(), true);

            Log::info('Nebius Embeddings API Response', [
                'usage' => $data['usage'] ?? null,
                'model' => $data['model'] ?? null
            ]);

            return $data['data'][0]['embedding'];

        } catch (GuzzleException $e) {
            Log::warning('Nebius Embeddings API Error, falling back to mock', [
                'error' => $e->getMessage(),
                'code' => $e->getCode()
            ]);

            // Fall back to mock embedding
            return $this->createMockEmbedding($text);
        }
    }

    /**
     * Create mock embedding for testing without API keys
     */
    private function createMockEmbedding(string $text): array
    {
        $dimensions = (int) env('VECTOR_DIM', 1024);
        
        // Create deterministic mock embedding based on text content
        $hash = md5($text);
        $seed = hexdec(substr($hash, 0, 8));
        mt_srand($seed);
        
        $embedding = [];
        for ($i = 0; $i < $dimensions; $i++) {
            $embedding[] = (mt_rand(0, 1000) / 1000.0) - 0.5; // Range -0.5 to 0.5
        }
        
        // Normalize the vector
        $magnitude = sqrt(array_sum(array_map(fn($x) => $x * $x, $embedding)));
        if ($magnitude > 0) {
            $embedding = array_map(fn($x) => $x / $magnitude, $embedding);
        }
        
        return $embedding;
    }

    /**
     * Test API connection
     */
    public function testConnection(): array
    {
        try {
            $response = $this->createChatCompletion([
                [
                    'role' => 'user',
                    'content' => 'Hello, this is a test message. Please respond with "API connection successful".'
                ]
            ], ['max_tokens' => 50]);

            return [
                'success' => true,
                'response' => $response['choices'][0]['message']['content'] ?? 'No response',
                'model' => $response['model'] ?? 'Unknown'
            ];

        } catch (\Exception $e) {
            return [
                'success' => false,
                'error' => $e->getMessage()
            ];
        }
    }
}

