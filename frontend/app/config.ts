// Where the backend runs. Set NEXT_PUBLIC_API_URL (for example in frontend/.env.local, or as
// a Docker build setting) to change it; by default it is the local backend.
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
