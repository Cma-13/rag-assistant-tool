"use client";

import { useState } from "react";

interface AuthModalProps {
  onClose: () => void;
  onAuthSuccess: (token: string, email: string) => void;
  initialMode?: "login" | "signup";
}

export default function AuthModal({ onClose, onAuthSuccess, initialMode = "login" }: AuthModalProps) {
  const [mode, setMode] = useState<"login" | "signup">(initialMode);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!email.trim() || !password.trim()) {
      setError("Please enter both email and password.");
      return;
    }
    setError("");
    setLoading(true);

    try {
      const endpoint = mode === "login" ? "/login" : "/signup";
      const res = await fetch(`http://127.0.0.1:8000${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();

      if (!res.ok) {
        setError(data.detail ?? "Something went wrong. Please try again.");
        setLoading(false);
        return;
      }

      onAuthSuccess(data.access_token, email);
    } catch {
      setError("Couldn't reach the server. Please try again.");
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4">
      <div className="w-full max-w-sm rounded-2xl bg-white p-6 shadow-xl">
        <div className="mb-5 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-[#1C1C1E]">
            {mode === "login" ? "Log in" : "Create an account"}
          </h2>
          <button
            onClick={onClose}
            className="text-[#9A968C] hover:text-[#1C1C1E]"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        <div className="space-y-3">
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded-xl border border-[#DDD9D0] px-3 py-2 text-[14.5px] outline-none focus:border-[#2D6A6A]"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
            className="w-full rounded-xl border border-[#DDD9D0] px-3 py-2 text-[14.5px] outline-none focus:border-[#2D6A6A]"
          />

          {error && <p className="text-[13px] text-red-600">{error}</p>}

          <button
            onClick={handleSubmit}
            disabled={loading}
            className="w-full rounded-xl bg-[#2D6A6A] py-2 text-[14.5px] font-medium text-white transition hover:bg-[#255757] disabled:opacity-50"
          >
            {loading
              ? "Please wait…"
              : mode === "login"
                ? "Log in"
                : "Sign up"}
          </button>
        </div>

        <p className="mt-4 text-center text-[13px] text-[#6B6B6B]">
          {mode === "login" ? (
            <>
              Don&apos;t have an account?{" "}
              <button
                onClick={() => {
                  setMode("signup");
                  setError("");
                }}
                className="font-medium text-[#2D6A6A] hover:underline"
              >
                Sign up
              </button>
            </>
          ) : (
            <>
              Already have an account?{" "}
              <button
                onClick={() => {
                  setMode("login");
                  setError("");
                }}
                className="font-medium text-[#2D6A6A] hover:underline"
              >
                Log in
              </button>
            </>
          )}
        </p>
      </div>
    </div>
  );
}