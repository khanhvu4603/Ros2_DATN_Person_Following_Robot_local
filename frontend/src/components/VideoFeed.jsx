import React, { useState } from 'react';
import { Camera, CameraOff } from 'lucide-react';

const VIDEO_URL = 'http://100.114.119.34:8000/video';


export const VideoFeed = ({ isRunning }) => {
    const [hasError, setHasError] = useState(false);

    return (
        <div className="w-full aspect-video bg-black rounded-3xl overflow-hidden shadow-2xl relative border-4 border-white/20">
            {isRunning && !hasError ? (
                <img
                    src={VIDEO_URL}
                    alt="Robot Camera Feed"
                    className="w-full h-full object-cover"
                    onError={() => setHasError(true)}
                />
            ) : (
                <div className="w-full h-full flex flex-col items-center justify-center text-gray-500 bg-gray-900">
                    {hasError ? (
                        <>
                            <CameraOff size={48} className="mb-4 opacity-50" />
                            <p className="text-sm font-medium">Stream Unavailable</p>
                        </>
                    ) : (
                        <>
                            <Camera size={48} className="mb-4 opacity-50" />
                            <p className="text-sm font-medium">Camera Offline</p>
                        </>
                    )}
                </div>
            )}

            {/* Overlay Status */}
            <div className="absolute top-4 left-4 px-3 py-1 bg-black/50 backdrop-blur-md rounded-full text-white text-xs font-medium flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isRunning ? 'bg-red-500 animate-pulse' : 'bg-gray-500'}`} />
                {isRunning ? 'LIVE' : 'OFFLINE'}
            </div>
        </div>
    );
};
