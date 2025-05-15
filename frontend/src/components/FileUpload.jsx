import React, { useState, useRef } from 'react';
import axios from 'axios';
import ResultModal from './ResultModal';
import VideoResultModal from './VideoResultModal';
import Loader from '../assets/Loading.gif';
import { API_ENDPOINTS } from '../config';

function FileUpload({ country }) {
  const [file, setFile] = useState(null);
  const [fileType, setFileType] = useState(null); // 'image' or 'video'
  const [loading, setLoading] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [videoModalOpen, setVideoModalOpen] = useState(false);
  const [resultData, setResultData] = useState(null);
  const [videoResultData, setVideoResultData] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (files) => {
    if (files && files[0]) {
      const selectedFile = files[0];
      setFile(selectedFile);
      setError(null);
      
      // Determine file type
      if (selectedFile.type.startsWith('image/')) {
        setFileType('image');
        setOriginalImage(URL.createObjectURL(selectedFile));
      } else if (selectedFile.type.startsWith('video/')) {
        setFileType('video');
        setOriginalImage(URL.createObjectURL(selectedFile));
      } else {
        setFileType(null);
        setOriginalImage(null);
        setError('Unsupported file type. Please upload an image or video.');
      }
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const files = event.dataTransfer.files;
    handleFileChange(files);
  };

  const handleSubmit = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('country', country);

    setLoading(true);
    setError(null);

    try {
      let response;
      
      if (fileType === 'image') {
        // Process image
        console.log(`Sending image to: ${API_ENDPOINTS.DETECT_LICENSE_PLATE}`);
        response = await axios.post(
          API_ENDPOINTS.DETECT_LICENSE_PLATE,
          formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
          }
        );
        
        console.log('Image processing response:', response.data);
        setResultData(response.data);
        setModalOpen(true);
      } else if (fileType === 'video') {
        // Process video
        console.log(`Sending video to: ${API_ENDPOINTS.PROCESS_VIDEO}`);
        
        // Add video-specific parameters
        formData.append('sample_rate', '5'); // Process every 5th frame
        formData.append('max_frames', '300'); // Process up to 300 frames
        
        response = await axios.post(
          API_ENDPOINTS.PROCESS_VIDEO,
          formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
            onUploadProgress: (progressEvent) => {
              const percentCompleted = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
              );
              console.log(`Upload progress: ${percentCompleted}%`);
            },
          }
        );
        
        console.log('Video processing response:', response.data);
        setVideoResultData(response.data);
        setVideoModalOpen(true);
      }
    } catch (error) {
      console.error('Error processing file:', error);
      setError(
        error.response?.data?.error || 
        error.response?.data?.message || 
        'Failed to process file. Please try again.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleCloseModal = () => {
    setModalOpen(false);
  };

  const handleCloseVideoModal = () => {
    setVideoModalOpen(false);
  };

  return (
    <>
      {loading && (
        <div className="absolute inset-0 flex flex-col justify-center items-center bg-black bg-opacity-50 z-50">
          <img src={Loader} alt="Loading..." className="w-28 h-28" />
          <p className="text-white mt-4">
            {fileType === 'video' ? 'Processing video... This may take a minute.' : 'Processing...'}
          </p>
        </div>
      )}
      <div className="p-4">
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          onClick={() => fileInputRef.current.click()}
          className="flex h-64 w-full cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed bg-gray-800 hover:bg-gray-700 dark:border-gray-500 dark:bg-gray-900 dark:hover:border-gray-400"
        >
          <div className="flex flex-col items-center justify-center pb-6 pt-5">
            <svg
              className="mb-4 h-8 w-8 text-gray-400 dark:text-gray-300"
              aria-hidden="true"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 20 16"
            >
              <path
                stroke="currentColor"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"
              />
            </svg>
            <p className="mb-2 text-sm text-gray-400 dark:text-gray-300">
              <span className="font-semibold">Click to upload</span> or drag and drop
            </p>
            <p className="text-xs text-gray-400 dark:text-gray-300">
              JPEG, PNG, JPG, or MP4
            </p>
          </div>
          <input
            ref={fileInputRef}
            id="file"
            type="file"
            accept="image/jpeg,image/png,image/jpg,video/mp4,video/avi,video/mov,video/mkv"
            className="hidden"
            onChange={(e) => handleFileChange(e.target.files)}
            multiple={false}
          />
        </div>

        {/* Show file preview */}
        {originalImage && (
          <div className="mt-4 flex items-center">
            {fileType === 'image' ? (
              <img
                src={originalImage}
                alt="Selected File Preview"
                className="w-12 h-12 object-cover rounded-sm"
              />
            ) : fileType === 'video' ? (
              <video
                src={originalImage}
                className="w-12 h-12 object-cover rounded-sm"
              />
            ) : null}
            <div className="ml-4 text-sm text-gray-600 dark:text-gray-300">
              {file.name} {/* Show the file name */}
              {fileType === 'video' && (
                <span className="ml-2 text-xs text-blue-400">
                  (Video will be processed frame by frame)
                </span>
              )}
            </div>
          </div>
        )}

        {/* Error message */}
        {error && (
          <div className="mt-4 text-center text-sm text-red-500 bg-red-100 p-2 rounded-md">
            {error}
          </div>
        )}

        <button
          disabled={!file || loading}
          onClick={handleSubmit}
          className="mt-4 w-full bg-primary text-white py-2 rounded-md hover:bg-[#592ac8] transition duration-300 disabled:opacity-50"
        >
          {!loading 
            ? fileType === 'video' 
              ? 'Process Video' 
              : 'Get Result' 
            : fileType === 'video'
              ? 'Processing Video...'
              : 'Processing...'}
        </button>

        {/* Image Result Modal */}
        <ResultModal
          isOpen={modalOpen}
          onClose={handleCloseModal}
          resultData={resultData}
          originalImage={originalImage}
        />

        {/* Video Result Modal */}
        <VideoResultModal 
          isOpen={videoModalOpen}
          onClose={handleCloseVideoModal}
          resultData={videoResultData}
          originalVideoSrc={originalImage}
        />
      </div>
    </>
  );
}

export default FileUpload;