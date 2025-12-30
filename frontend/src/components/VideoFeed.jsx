import React, { useState, useRef, useEffect } from 'react';
import { Camera, CameraOff } from 'lucide-react';

export const VideoFeed = ({ isConnected }) => {
    const imgRef = useRef(null);
    const wsRef = useRef(null);

    useEffect(() => {
        if (!isConnected) {
            if (wsRef.current) {
                wsRef.current.close();
                wsRef.current = null;
            }
            return;
        }

        // Connect to Video WebSocket
        const wsUrl = `ws://${window.location.hostname}:8000/ws/video`;
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
            console.log("Connected to Video WebSocket");
        };

        ws.onmessage = (event) => {
            if (imgRef.current) {
                // Revoke previous URL to avoid memory leak
                if (imgRef.current.src) {
                    URL.revokeObjectURL(imgRef.current.src);
                }
                // Create new URL from Blob
                const blob = event.data;
                const url = URL.createObjectURL(blob);
                imgRef.current.src = url;
            }
        };

        ws.onerror = (error) => {
            console.error("Video WebSocket error:", error);
        };

        return () => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
        };
    }, [isConnected]);

    return (
        <div className="w-full aspect-video bg-black rounded-3xl overflow-hidden shadow-2xl relative border-4 border-white/20 group">
            <div className="absolute inset-0 flex items-center justify-center">
                {isConnected ? (
                    <img
                        ref={imgRef}
                        alt="Robot Feed"
                        className="w-full h-full object-contain"
                    />
                ) : (
                    <div className="flex flex-col items-center gap-3 text-gray-500">
                        <div className="w-12 h-12 rounded-full bg-gray-800 flex items-center justify-center">
                            <div className="w-6 h-6 border-2 border-gray-600 border-t-transparent rounded-full animate-spin" />
                        </div>
                        <span className="text-sm font-medium">Connecting to feed...</span>
                    </div>
                )}
            </div>

            {/* Overlay Status */}
            <div className="absolute top-4 left-4 px-3 py-1 bg-black/50 backdrop-blur-md rounded-full text-white text-xs font-medium flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-red-500 animate-pulse' : 'bg-gray-500'}`} />
                {isConnected ? 'LIVE' : 'OFFLINE'}
            </div>
        </div>
    );
};
