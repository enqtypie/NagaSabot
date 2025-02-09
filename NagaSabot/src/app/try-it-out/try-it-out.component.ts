import { Component, ElementRef, ViewChild, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PermissionService } from '../../app/permission.service';

interface VideoData {
  file: File;
  url: string;
  timestamp: number;
}

@Component({
  selector: 'app-try-it-out',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './try-it-out.component.html'
})
export class TryItOutComponent implements OnDestroy {
  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;
  
  // State management
  isRecording = false;
  showCameraModal = false;
  isProcessing = false;
  errorMessage: string | null = null;
  
  // Media handling
  private mediaStream: MediaStream | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private recordedChunks: Blob[] = [];
  private recordingStartTime: number = 0;
  private maxRecordingDuration = 300000; // 5 minutes in milliseconds
  private recordingTimer: any;

  constructor(private permissionService: PermissionService) {}

  ngOnDestroy() {
    this.cleanup();
  }

  private cleanup() {
    this.stopRecording();
    this.clearError();
    this.resetState();
  }

  private resetState() {
    this.isRecording = false;
    this.isProcessing = false;
    this.recordedChunks = [];
    this.recordingStartTime = 0;
    if (this.recordingTimer) {
      clearTimeout(this.recordingTimer);
      this.recordingTimer = null;
    }
  }

  private clearError() {
    this.errorMessage = null;
  }

  // Modal Management
  openCameraModal() {
    this.showCameraModal = true;
    this.clearError();
    this.requestCameraAccess();
  }

  closeCameraModal() {
    this.cleanup();
    this.showCameraModal = false;
  }

  // Camera Permissions and Setup
  private async requestCameraAccess(): Promise<boolean> {
    try {
      if (!this.permissionService.hasCameraPermission()) {
        const granted = await this.permissionService.requestCameraPermission();
        if (!granted) {
          this.errorMessage = 'Camera permission is required for recording.';
          return false;
        }
      }
      return true;
    } catch (error) {
      this.handleError('Failed to get camera permission', error);
      return false;
    }
  }

  // Recording Management
  async startCamera() {
    if (this.isRecording || this.isProcessing) return;
    
    try {
      const hasPermission = await this.requestCameraAccess();
      if (!hasPermission) return;

      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: true
      });

      this.videoElement.nativeElement.srcObject = this.mediaStream;
      await this.videoElement.nativeElement.play();
      
      this.setupRecording();
    } catch (error) {
      this.handleError('Failed to start camera', error);
    }
  }

  private setupRecording() {
    if (!this.mediaStream) return;

    try {
      this.mediaRecorder = new MediaRecorder(this.mediaStream, {
        mimeType: 'video/webm;codecs=vp8,opus'
      });

      this.mediaRecorder.ondataavailable = this.handleDataAvailable.bind(this);
      this.mediaRecorder.onstart = this.handleRecordingStart.bind(this);
      this.mediaRecorder.onstop = this.handleRecordingStop.bind(this);
      
      // Fixed: Handle the error event with the DOMException
      this.mediaRecorder.onerror = (event: Event) => {
        const error = event instanceof ErrorEvent ? event.error : new DOMException('Unknown recording error');
        this.handleError('Recording error', error);
        this.stopRecording();
      };

      this.mediaRecorder.start(1000); // Collect data every second
      this.isRecording = true;

      // Set maximum recording duration
      this.recordingTimer = setTimeout(() => {
        if (this.isRecording) {
          this.stopRecording();
        }
      }, this.maxRecordingDuration);

    } catch (error) {
      this.handleError('Failed to setup recording', error);
    }
  }

  stopCamera() {
    this.stopRecording();
  }

  private stopRecording() {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    if (this.videoElement?.nativeElement) {
      this.videoElement.nativeElement.srcObject = null;
    }

    this.isRecording = false;
    if (this.recordingTimer) {
      clearTimeout(this.recordingTimer);
      this.recordingTimer = null;
    }
  }

  // Recording Event Handlers
  private handleDataAvailable(event: BlobEvent) {
    if (event.data && event.data.size > 0) {
      this.recordedChunks.push(event.data);
    }
  }

  private handleRecordingStart() {
    this.recordingStartTime = Date.now();
    this.isRecording = true;
    this.clearError();
  }

  private handleRecordingStop() {
    this.isProcessing = true;
    try {
      const blob = new Blob(this.recordedChunks, { type: 'video/webm' });
      const videoData: VideoData = {
        file: new File([blob], `recording-${Date.now()}.webm`, { type: 'video/webm' }),
        url: URL.createObjectURL(blob),
        timestamp: Date.now()
      };
      
      this.processVideoData(videoData);
    } catch (error) {
      this.handleError('Failed to process recording', error);
    } finally {
      this.isProcessing = false;
    }
  }

  // File Upload Handling
  async handleFileSelection(event: Event) {
    this.clearError();
    this.isProcessing = true;
    
    try {
      const input = event.target as HTMLInputElement;
      if (!input.files || !input.files[0]) {
        throw new Error('No file selected');
      }

      const file = input.files[0];
      if (!file.type.startsWith('video/')) {
        throw new Error('Please select a valid video file');
      }

      const videoData: VideoData = {
        file: file,
        url: URL.createObjectURL(file),
        timestamp: Date.now()
      };

      await this.processVideoData(videoData);
    } catch (error) {
      this.handleError('Failed to process video file', error);
    } finally {
      this.isProcessing = false;
      if (this.fileInput) {
        this.fileInput.nativeElement.value = '';
      }
    }
  }

  // Video Processing
  private async processVideoData(videoData: VideoData) {
    try {
      // Here you would implement your video processing logic
      // For example, sending to a server or processing locally
      console.log('Processing video:', videoData);
      
      // Example of processing steps:
      // 1. Validate video duration
      const duration = await this.getVideoDuration(videoData.url);
      if (duration > this.maxRecordingDuration) {
        throw new Error('Video duration exceeds maximum allowed length');
      }

      // 2. Check file size
      if (videoData.file.size > 100 * 1024 * 1024) { // 100MB limit
        throw new Error('File size exceeds maximum allowed size');
      }

      // 3. Process the video
      // await this.yourVideoProcessingService.process(videoData);

    } catch (error) {
      this.handleError('Video processing failed', error);
      throw error;
    }
  }

  private getVideoDuration(url: string): Promise<number> {
    return new Promise((resolve, reject) => {
      const video = document.createElement('video');
      video.onloadedmetadata = () => resolve(video.duration * 1000);
      video.onerror = () => reject('Error loading video');
      video.src = url;
    });
  }

  // Error Handling
  private handleError(message: string, error: any) {
    console.error(message, error);
    this.errorMessage = `${message}: ${error.message || 'Unknown error occurred'}`;
  }
}