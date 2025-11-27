<?php

namespace App\Http\Controllers;

use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;
use Illuminate\Support\Facades\Validator;
use Illuminate\Validation\Rule;

class AdminUserController extends Controller
{
    /**
     * Display a listing of users
     */
    public function index(Request $request)
    {
        $limit = (int) min((int) $request->query('limit', 50), 100);
        $search = $request->query('search');
        
        $query = User::orderByDesc('id');
        
        if ($search) {
            $query->where(function($q) use ($search) {
                $q->where('name', 'ILIKE', "%{$search}%")
                  ->orWhere('email', 'ILIKE', "%{$search}%")
                  ->orWhere('full_name', 'ILIKE', "%{$search}%");
            });
        }
        
        $users = $query->limit($limit)->get([
            'id', 'name', 'full_name', 'email', 'join_date', 'is_active', 'created_at'
        ]);
        
        return response()->json([
            'users' => $users->map(function($user) {
                return [
                    'id' => $user->id,
                    'name' => $user->name,
                    'full_name' => $user->full_name,
                    'email' => $user->email,
                    'join_date' => $user->join_date,
                    'is_active' => $user->is_active,
                    'created_at' => $user->created_at->toISOString(),
                ];
            })
        ]);
    }

    /**
     * Show the form for creating a new user
     */
    public function show($id)
    {
        $user = User::findOrFail($id);
        
        return response()->json([
            'user' => [
                'id' => $user->id,
                'name' => $user->name,
                'full_name' => $user->full_name,
                'email' => $user->email,
                'password_hash' => $user->password, // Include password hash for admin viewing
                'join_date' => $user->join_date,
                'is_active' => $user->is_active,
                'progress' => $user->progress,
                'preferences' => $user->preferences,
                'created_at' => $user->created_at->toISOString(),
                'updated_at' => $user->updated_at->toISOString(),
            ]
        ]);
    }

    /**
     * Store a newly created user
     */
    public function store(Request $request)
    {
        $validator = Validator::make($request->all(), [
            'name' => ['required', 'string', 'max:255'],
            'full_name' => ['required', 'string', 'max:255'],
            'email' => ['required', 'string', 'email', 'max:255', 'unique:users'],
            'password' => ['required', 'string', 'min:8'],
            'is_active' => ['boolean'],
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            $user = new User();
            $user->name = $request->name;
            $user->full_name = $request->full_name;
            $user->email = $request->email;
            $user->password = Hash::make($request->password);
            $user->join_date = $request->join_date ?? now()->toDateString();
            $user->is_active = $request->is_active ?? 1;
            $user->save();
            
            // Set default progress and preferences
            $user->progress = $user->getDefaultProgress();
            $user->preferences = $user->getDefaultPreferences();
            $user->save();

            return response()->json([
                'message' => 'User created successfully',
                'user' => [
                    'id' => $user->id,
                    'name' => $user->name,
                    'full_name' => $user->full_name,
                    'email' => $user->email,
                    'join_date' => $user->join_date,
                    'is_active' => $user->is_active,
                    'created_at' => $user->created_at->toISOString(),
                ]
            ], 201);

        } catch (\Exception $e) {
            return response()->json([
                'message' => 'User creation failed',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Update the specified user
     */
    public function update(Request $request, $id)
    {
        $user = User::findOrFail($id);
        
        $validator = Validator::make($request->all(), [
            'name' => ['sometimes', 'string', 'max:255'],
            'full_name' => ['sometimes', 'string', 'max:255'],
            'email' => ['sometimes', 'string', 'email', 'max:255', Rule::unique('users')->ignore($user->id)],
            'password' => ['sometimes', 'string', 'min:8'],
            'is_active' => ['sometimes', 'boolean'],
            'join_date' => ['sometimes', 'date'],
            'progress' => ['sometimes', 'array'],
            'preferences' => ['sometimes', 'array'],
        ]);

        if ($validator->fails()) {
            return response()->json([
                'message' => 'Validation failed',
                'errors' => $validator->errors()
            ], 422);
        }

        try {
            $updateData = [];
            
            if ($request->has('name')) {
                $updateData['name'] = $request->name;
            }
            
            if ($request->has('full_name')) {
                $updateData['full_name'] = $request->full_name;
            }
            
            if ($request->has('email')) {
                $updateData['email'] = $request->email;
            }
            
            if ($request->has('password')) {
                $updateData['password'] = Hash::make($request->password);
            }
            
            if ($request->has('is_active')) {
                $updateData['is_active'] = $request->is_active ? 1 : 0;
            }
            
            if ($request->has('join_date')) {
                $updateData['join_date'] = $request->join_date;
            }
            
            if ($request->has('progress')) {
                $updateData['progress'] = $request->progress;
            }
            
            if ($request->has('preferences')) {
                $updateData['preferences'] = $request->preferences;
            }

            $user->update($updateData);

            return response()->json([
                'message' => 'User updated successfully',
                'user' => [
                    'id' => $user->id,
                    'name' => $user->name,
                    'full_name' => $user->full_name,
                    'email' => $user->email,
                    'join_date' => $user->join_date,
                    'is_active' => $user->is_active,
                    'created_at' => $user->created_at->toISOString(),
                    'updated_at' => $user->updated_at->toISOString(),
                ]
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'message' => 'User update failed',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Remove the specified user from database
     */
    public function destroy($id)
    {
        try {
            $user = User::findOrFail($id);
            $userName = $user->name;
            $userEmail = $user->email;
            
            // Delete the user from database
            $user->delete();

            return response()->json([
                'message' => "User '{$userName}' ({$userEmail}) has been permanently deleted from the database",
                'deleted_user' => [
                    'id' => $id,
                    'name' => $userName,
                    'email' => $userEmail,
                ]
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'User not found'
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'User deletion failed',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Toggle user active status
     */
    public function toggleStatus($id)
    {
        try {
            $user = User::findOrFail($id);
            $user->is_active = $user->is_active ? 0 : 1;
            $user->save();

            return response()->json([
                'message' => 'User status updated successfully',
                'user' => [
                    'id' => $user->id,
                    'name' => $user->name,
                    'email' => $user->email,
                    'is_active' => $user->is_active,
                ]
            ]);

        } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
            return response()->json([
                'message' => 'User not found'
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Status update failed',
                'error' => $e->getMessage()
            ], 500);
        }
    }

    /**
     * Get user statistics
     */
    public function stats()
    {
        try {
            $totalUsers = User::count();
            $activeUsers = User::where('is_active', 1)->count();
            $inactiveUsers = User::where('is_active', 0)->count();
            $recentUsers = User::where('created_at', '>=', now()->subDays(7))->count();

            return response()->json([
                'stats' => [
                    'total' => $totalUsers,
                    'active' => $activeUsers,
                    'inactive' => $inactiveUsers,
                    'recent' => $recentUsers,
                ]
            ]);

        } catch (\Exception $e) {
            return response()->json([
                'message' => 'Failed to get user statistics',
                'error' => $e->getMessage()
            ], 500);
        }
    }
}