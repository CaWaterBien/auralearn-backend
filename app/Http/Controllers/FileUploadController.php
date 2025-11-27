<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Facades\Validator;
use Illuminate\Support\Str;
use Exception;

class FileUploadController extends Controller
{
    /**
     * Upload an image file
     */
    public function uploadImage(Request $request): JsonResponse
    {
        try {
            $validator = Validator::make($request->all(), [
                'image' => [
                    'required',
                    'file',
                    'mimes:jpeg,jpg,png,gif,webp',
                    'max:10240' // 10MB max
                ],
                'type' => 'required|string|in:topic_image,lesson_image,course_image'
            ]);

            if ($validator->fails()) {
                return response()->json([
                    'success' => false,
                    'message' => 'Validation failed',
                    'errors' => $validator->errors()
                ], 422);
            }

            $file = $request->file('image');
            $type = $request->input('type', 'topic_image');

            // Create directory structure
            $directory = $this->getUploadDirectory('images', $type);
            
            // Generate unique filename
            $filename = $this->generateUniqueFilename($file);
            
            // Store file
            $path = $file->storeAs($directory, $filename, 'public');
            
            if (!$path) {
                throw new Exception('Failed to store file');
            }

            // Get full URL for the uploaded file
            $storageUrl = Storage::url($path);
            $fullUrl = 'http://localhost:8000' . $storageUrl;
            
            // Get file info
            $fileInfo = [
                'original_name' => $file->getClientOriginalName(),
                'filename' => $filename,
                'path' => $path,
                'file_path' => $fullUrl,
                'size' => $file->getSize(),
                'mime_type' => $file->getMimeType(),
                'type' => $type,
                'uploaded_at' => now()->toISOString()
            ];

            return response()->json([
                'success' => true,
                'message' => 'Image uploaded successfully',
                'file_path' => $fullUrl,
                'file_info' => $fileInfo
            ]);

        } catch (Exception $e) {
            return response()->json([
                'success' => false,
                'message' => 'Upload failed: ' . $e->getMessage()
            ], 500);
        }
    }

    /**
     * Upload a video file
     */
    public function uploadVideo(Request $request): JsonResponse
    {
        try {
            $validator = Validator::make($request->all(), [
                'video' => [
                    'required',
                    'file',
                    'mimes:mp4,avi,mov,wmv,flv,webm',
                    'max:102400' // 100MB max
                ],
                'type' => 'required|string|in:topic_video,lesson_video,course_video'
            ]);

            if ($validator->fails()) {
                return response()->json([
                    'success' => false,
                    'message' => 'Validation failed',
                    'errors' => $validator->errors()
                ], 422);
            }

            $file = $request->file('video');
            $type = $request->input('type', 'topic_video');

            // Create directory structure
            $directory = $this->getUploadDirectory('videos', $type);
            
            // Generate unique filename
            $filename = $this->generateUniqueFilename($file);
            
            // Store file
            $path = $file->storeAs($directory, $filename, 'public');
            
            if (!$path) {
                throw new Exception('Failed to store file');
            }

            // Get full URL for the uploaded file
            $storageUrl = Storage::url($path);
            $fullUrl = 'http://localhost:8000' . $storageUrl;
            
            // Get file info
            $fileInfo = [
                'original_name' => $file->getClientOriginalName(),
                'filename' => $filename,
                'path' => $path,
                'file_path' => $fullUrl,
                'size' => $file->getSize(),
                'mime_type' => $file->getMimeType(),
                'type' => $type,
                'uploaded_at' => now()->toISOString()
            ];

            return response()->json([
                'success' => true,
                'message' => 'Video uploaded successfully',
                'file_path' => $fullUrl,
                'file_info' => $fileInfo
            ]);

        } catch (Exception $e) {
            return response()->json([
                'success' => false,
                'message' => 'Upload failed: ' . $e->getMessage()
            ], 500);
        }
    }

    /**
     * Delete an uploaded file
     */
    public function deleteFile(Request $request): JsonResponse
    {
        try {
            $validator = Validator::make($request->all(), [
                'file_path' => 'required|string'
            ]);

            if ($validator->fails()) {
                return response()->json([
                    'success' => false,
                    'message' => 'Validation failed',
                    'errors' => $validator->errors()
                ], 422);
            }

            $filePath = $request->input('file_path');
            
            // Extract the storage path from the URL
            $storagePath = str_replace('/storage/', '', $filePath);
            
            if (Storage::disk('public')->exists($storagePath)) {
                Storage::disk('public')->delete($storagePath);
                
                return response()->json([
                    'success' => true,
                    'message' => 'File deleted successfully'
                ]);
            }

            return response()->json([
                'success' => false,
                'message' => 'File not found'
            ], 404);

        } catch (Exception $e) {
            return response()->json([
                'success' => false,
                'message' => 'Delete failed: ' . $e->getMessage()
            ], 500);
        }
    }

    /**
     * Get upload directory based on file type and category
     */
    private function getUploadDirectory(string $fileType, string $category): string
    {
        $year = date('Y');
        $month = date('m');
        
        return "uploads/{$fileType}/{$category}/{$year}/{$month}";
    }

    /**
     * Generate unique filename while preserving extension
     */
    private function generateUniqueFilename($file): string
    {
        $extension = $file->getClientOriginalExtension();
        $originalName = pathinfo($file->getClientOriginalName(), PATHINFO_FILENAME);
        
        // Clean the original name
        $cleanName = Str::slug(substr($originalName, 0, 50));
        
        // Generate unique suffix
        $uniqueId = Str::random(8);
        $timestamp = time();
        
        return "{$cleanName}_{$timestamp}_{$uniqueId}.{$extension}";
    }

    /**
     * Get file info
     */
    public function getFileInfo(Request $request): JsonResponse
    {
        try {
            $validator = Validator::make($request->all(), [
                'file_path' => 'required|string'
            ]);

            if ($validator->fails()) {
                return response()->json([
                    'success' => false,
                    'message' => 'Validation failed',
                    'errors' => $validator->errors()
                ], 422);
            }

            $filePath = $request->input('file_path');
            $storagePath = str_replace('/storage/', '', $filePath);
            
            if (Storage::disk('public')->exists($storagePath)) {
                $size = Storage::disk('public')->size($storagePath);
                $lastModified = Storage::disk('public')->lastModified($storagePath);
                
                return response()->json([
                    'success' => true,
                    'file_info' => [
                        'path' => $filePath,
                        'size' => $size,
                        'size_formatted' => $this->formatFileSize($size),
                        'last_modified' => date('Y-m-d H:i:s', $lastModified),
                        'exists' => true
                    ]
                ]);
            }

            return response()->json([
                'success' => false,
                'message' => 'File not found',
                'file_info' => [
                    'exists' => false
                ]
            ], 404);

        } catch (Exception $e) {
            return response()->json([
                'success' => false,
                'message' => 'Failed to get file info: ' . $e->getMessage()
            ], 500);
        }
    }

    /**
     * Format file size in human readable format
     */
    private function formatFileSize(int $size): string
    {
        $units = ['B', 'KB', 'MB', 'GB'];
        $unit = 0;
        
        while ($size >= 1024 && $unit < 3) {
            $size /= 1024;
            $unit++;
        }
        
        return round($size, 2) . ' ' . $units[$unit];
    }
}
