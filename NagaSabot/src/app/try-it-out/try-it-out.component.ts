import { Component, ViewChild, ElementRef, OnDestroy, PLATFORM_ID, Inject, AfterViewInit } from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { VideoResultComponent } from '../video-result/video-result.component';
import { VideoService } from '../services/video.service';
import { HeaderComponent } from '../header/header.component';

@Component({
  selector: 'app-try-it-out',
  standalone: true,
  imports: [CommonModule, VideoResultComponent, HeaderComponent],
  templateUrl: './try-it-out.component.html'
})
export class TryItOutComponent implements OnDestroy, AfterViewInit {
  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;
  @ViewChild('fileInput') fileInput!: ElementRef<HTMLInputElement>;
  @ViewChild('canvas') canvas!: ElementRef<HTMLCanvasElement>;

  showCameraModal = false;
  isRecording = false;
  isModalClosing = false;
  videoBlob: Blob | null = null;
  areLipsVisible = false;
  private mediaStream: MediaStream | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];
  private faceLandmarker: any = null;
  private animationFrameId: number | null = null;
  private isViewInitialized = false;
  private lastVideoTime = -1;
  private noLipsDetectedCount = 0;
  private readonly NO_LIPS_THRESHOLD = 10;
  
  readonly REQUIRED_FRAMES = 30;
  currentFrameCount = 0;
  isFrameCollectionComplete = false;
  canvasWidth = 640;
  canvasHeight = 360;
  predictedPhrase: string | null = null;
  predictionConfidence: number | null = null;
  isLoading = false;
  errorMessage: string | null = null;

  constructor(
    private videoService: VideoService,
    @Inject(PLATFORM_ID) private platformId: Object
  ) {}

  ngAfterViewInit() {
    if (isPlatformBrowser(this.platformId)) {
      this.isViewInitialized = true;
      this.initializeFaceLandmarker();
    }
  }

  private async initializeFaceLandmarker() {
    if (!isPlatformBrowser(this.platformId)) return;

    try {
      const { FaceLandmarker, FilesetResolver, DrawingUtils } = await import('@mediapipe/tasks-vision');
      
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
      );
      
      this.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
          delegate: "CPU"
        },
        outputFaceBlendshapes: true,
        runningMode: "VIDEO",
        numFaces: 1
      });

      (this as any).FaceLandmarker = FaceLandmarker;
      (this as any).DrawingUtils = DrawingUtils;
    } catch (error) {
      console.error('Error initializing face landmarker:', error);
    }
  }

  async openCameraModal() {
    if (!isPlatformBrowser(this.platformId)) {
      console.error('Camera access is only available in browser environments');
      return;
    }

    // Reset all relevant state for a new session
    this.currentFrameCount = 0;
    this.isFrameCollectionComplete = false;
    this.areLipsVisible = false;
    this.noLipsDetectedCount = 0;
    this.lastVideoTime = -1;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    if (this.faceLandmarker) {
      this.faceLandmarker = null;
    }

    this.showCameraModal = true;

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('getUserMedia is not supported in this browser');
      }

      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');

      this.mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 360 },
          frameRate: { ideal: 120, max: 240 }, // Request maximum possible FPS
          facingMode: 'user',
          deviceId: videoDevices[0]?.deviceId
        }
      });

      await this.waitForViewInitialization();
      if (!this.faceLandmarker) {
        await this.initializeFaceLandmarker();
      }
      this.videoElement.nativeElement.srcObject = this.mediaStream;
      
      await new Promise<void>((resolve) => {
        this.videoElement.nativeElement.onloadedmetadata = () => {
          this.videoElement.nativeElement.play();
          this.updateCanvasSize();
          window.addEventListener('resize', this.updateCanvasSize);
          window.addEventListener('orientationchange', this.updateCanvasSize);
          resolve();
          this.startFaceTracking(); // Only start face tracking, not recording
        };
      });
    } catch (err: any) {
      console.error('Camera error:', err);
      alert(this.getErrorMessage(err));
      this.closeCameraModal();
    }
  }

  private async waitForViewInitialization() {
    return new Promise<void>(resolve => {
      const checkView = () => {
        if (this.isViewInitialized && this.videoElement && this.canvas) {
          resolve();
        } else {
          setTimeout(checkView, 100);
        }
      };
      checkView();
    });
  }

  private async startFaceTracking() {
    // Cancel any previous animation frame loop
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    if (!this.faceLandmarker || !this.videoElement?.nativeElement || !this.canvas?.nativeElement) return;

    const ctx = this.canvas.nativeElement.getContext('2d');
    if (!ctx) return;

    const drawingUtils = new (this as any).DrawingUtils(ctx);
    const FaceLandmarker = (this as any).FaceLandmarker;

    const detectFaces = async () => {
      try {
        if (this.videoElement.nativeElement.paused || this.videoElement.nativeElement.ended) {
          await this.videoElement.nativeElement.play();
        }

        // Only process if video time has changed
        if (this.videoElement.nativeElement.currentTime !== this.lastVideoTime) {
          this.lastVideoTime = this.videoElement.nativeElement.currentTime;
          const results = this.faceLandmarker.detectForVideo(
            this.videoElement.nativeElement,
            performance.now()
          );

          ctx.clearRect(0, 0, this.canvas.nativeElement.width, this.canvas.nativeElement.height);

          if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            // Reset the no lips detected counter when lips are found
            this.noLipsDetectedCount = 0;
            this.areLipsVisible = true;

            // Increment frame count if recording and lips are visible
            if (this.isRecording && !this.isFrameCollectionComplete) {
              this.currentFrameCount++;
              if (this.currentFrameCount >= this.REQUIRED_FRAMES) {
                this.isFrameCollectionComplete = true;
                this.stopRecording();
                return; // Exit the detection loop after stopping
              }
            }

            for (const landmarks of results.faceLandmarks) {
              // Create a mirrored version of the landmarks
              const mirroredLandmarks = landmarks.map((point: { x: number; y: number; z: number }) => ({
                ...point,
                x: 1 - point.x // Mirror the x coordinate
              }));

              // Draw lip connectors with mirrored landmarks
              drawingUtils.drawConnectors(
                mirroredLandmarks,
                FaceLandmarker.FACE_LANDMARKS_LIPS,
                { color: "#FFFFFF", lineWidth: 2 }
              );

              // Draw lip points with mirrored landmarks
              const lipPoints = FaceLandmarker.FACE_LANDMARKS_LIPS.flat();
              for (const index of lipPoints) {
                const point = mirroredLandmarks[index];
                if (!point) continue; // Skip if undefined
                ctx.beginPath();
                ctx.arc(
                  point.x * this.canvas.nativeElement.width,
                  point.y * this.canvas.nativeElement.height,
                  2,
                  0,
                  2 * Math.PI
                );
                ctx.fillStyle = '#FF3030';
                ctx.fill();
              }
            }
          } else {
            // Increment the counter when no lips are detected
            this.noLipsDetectedCount++;
            
            // If we haven't detected lips for several consecutive frames, consider them not visible
            if (this.noLipsDetectedCount >= this.NO_LIPS_THRESHOLD) {
              this.areLipsVisible = false;
            }
          }
        }
      } catch (error) {
        console.error('Face tracking error:', error);
      }

      if (!this.isFrameCollectionComplete) {
        this.animationFrameId = requestAnimationFrame(detectFaces);
      }
    };

    detectFaces();
  }

  private getErrorMessage(error: any): string {
    const messages: { [key: string]: string } = {
      'NotAllowedError': 'Please make sure you have granted camera permissions.',
      'NotFoundError': 'No camera device found.',
      'NotReadableError': 'Camera is in use by another application.',
      'OverconstrainedError': 'Camera does not support the requested constraints.',
      'StreamApiNotSupportedError': 'Stream API is not supported in this browser.'
    };

    return `Unable to access camera. ${messages[error.name] || `Error: ${error.message || 'Unknown error occurred'}`}`;
  }

  closeCameraModal() {
    this.isModalClosing = true;
    setTimeout(() => {
      this.showCameraModal = false;
      this.isModalClosing = false;
      this.stopCamera();
      window.removeEventListener('resize', this.updateCanvasSize);
      window.removeEventListener('orientationchange', this.updateCanvasSize);
    }, 300);
  }

  startRecording() {
    if (!this.mediaStream) return;

    this.chunks = [];
    this.isRecording = true;
    this.currentFrameCount = 0;
    this.isFrameCollectionComplete = false;
    
    // Configure MediaRecorder with quality settings
    this.mediaRecorder = new MediaRecorder(this.mediaStream, {
      mimeType: 'video/webm;codecs=vp9',
      videoBitsPerSecond: 2500000 // 2.5 Mbps
    });

    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        this.chunks.push(event.data);
        console.log('Chunk received, size:', event.data.size);
      }
    };

    this.mediaRecorder.onstop = () => {
      this.handleRecordingComplete();
    };

    // Start MediaRecorder with no interval for best FPS
    this.mediaRecorder.start();
    // Log the actual FPS achieved (optional, for debugging)
    let lastFrameCount = 0;
    let lastTime = Date.now();
    const fpsLogger = setInterval(() => {
      if (!this.isRecording) { clearInterval(fpsLogger); return; }
      const now = Date.now();
      const fps = (this.currentFrameCount - lastFrameCount) / ((now - lastTime) / 1000);
      console.log(`Approx. FPS: ${fps.toFixed(2)}`);
      lastFrameCount = this.currentFrameCount;
      lastTime = now;
    }, 1000);
    console.log('Recording started with settings:', {
      mimeType: this.mediaRecorder.mimeType,
      videoBitsPerSecond: this.mediaRecorder.videoBitsPerSecond,
      state: this.mediaRecorder.state
    });
  }

  stopRecording() {
    if (!this.isRecording) return; // Guard: only stop once
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
      this.isRecording = false;
      console.log('Recording stopped, waiting for blob...');
      if (this.showCameraModal) {
        this.closeCameraModal();
      }
    }
  }

  private handleRecordingComplete() {
    console.log('Recording complete, creating blob...');
    this.videoBlob = new Blob(this.chunks, { type: 'video/webm' });
    console.log('Video blob created:', this.videoBlob);
    console.log('Video blob size:', this.videoBlob.size);
    if (this.videoBlob) {
      this.isLoading = true;
      this.errorMessage = null;
      this.checkAndPredictVideo(new File([this.videoBlob], 'recorded.webm', { type: 'video/webm' }));
    }
  }

  stopCamera() {
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    if (this.videoElement?.nativeElement) {
      this.videoElement.nativeElement.srcObject = null;
    }

    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    if (this.faceLandmarker) {
      this.faceLandmarker = null;
    }
  }

  handleFileSelection(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      console.log('File selected:', input.files[0]);
      console.log('File size:', input.files[0].size);
      this.videoBlob = input.files[0];
      this.isLoading = true;
      this.errorMessage = null;
      this.checkAndPredictVideo(input.files[0]);
    }
  }

  private checkAndPredictVideo(file: File) {
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.src = URL.createObjectURL(file);
    video.onloadedmetadata = () => {
      URL.revokeObjectURL(video.src);
      if (video.duration < 1) {
        this.isLoading = false;
        this.errorMessage = 'Please record or upload a longer video (at least 1 second) for best results.';
        return;
      }
      this.predictVideo(file);
    };
  }

  private predictVideo(file: File) {
    this.videoService.predictLipreading(file)
      .subscribe(result => {
        this.predictedPhrase = result.phrase;
        this.predictionConfidence = result.confidence;
        this.isLoading = false;
        this.errorMessage = null;
      }, err => {
        this.predictedPhrase = 'Prediction failed';
        this.predictionConfidence = null;
        this.isLoading = false;
        this.errorMessage = 'Prediction failed. Please try again with a clearer or longer video.';
      });
  }

  handleRestart() {
    console.log('Handling restart...');
    if (this.videoBlob) {
      console.log('Cleaning up video blob');
      this.videoBlob = null;
    }
    if (this.fileInput) {
      this.fileInput.nativeElement.value = '';
    }
    // Reset all relevant state before opening camera modal
    this.currentFrameCount = 0;
    this.isFrameCollectionComplete = false;
    this.areLipsVisible = false;
    this.noLipsDetectedCount = 0;
    this.lastVideoTime = -1;
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    if (this.faceLandmarker) {
      this.faceLandmarker = null;
    }
    // Reopen camera modal (recording will start on button click)
    this.openCameraModal();
  }

  ngOnDestroy() {
    this.stopCamera();
  }

  updateCanvasSize = () => {
    if (this.videoElement && this.videoElement.nativeElement) {
      const rect = this.videoElement.nativeElement.getBoundingClientRect();
      this.canvasWidth = Math.floor(rect.width);
      this.canvasHeight = Math.floor(rect.height);
    }
  }
}