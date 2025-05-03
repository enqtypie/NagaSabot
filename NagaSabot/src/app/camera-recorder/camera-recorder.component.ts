import { Component, ViewChild, ElementRef, OnDestroy, PLATFORM_ID, Inject, AfterViewInit, Output, EventEmitter } from '@angular/core';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { VideoService } from '../services/video.service';

@Component({
  selector: 'app-camera-recorder',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './camera-recorder.component.html'
})
export class CameraRecorderComponent implements OnDestroy, AfterViewInit {
  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;
  @ViewChild('canvas') canvas!: ElementRef<HTMLCanvasElement>;

  @Output() videoRecorded = new EventEmitter<Blob>();
  @Output() close = new EventEmitter<void>();

  showCameraModal = true;
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
  
  readonly REQUIRED_FRAMES = 75;
  currentFrameCount = 0;
  isFrameCollectionComplete = false;
  canvasWidth = 640;
  canvasHeight = 480;
  predictedPhrase: string | null = null;
  predictionConfidence: number | null = null;
  isLoading = false;
  errorMessage: string | null = null;
  frameCount = 0;

  constructor(
    private videoService: VideoService,
    @Inject(PLATFORM_ID) private platformId: Object
  ) {}

  ngAfterViewInit() {
    if (isPlatformBrowser(this.platformId)) {
      this.isViewInitialized = true;
      this.initializeFaceLandmarker();
      this.openCameraModal();
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
          height: { ideal: 480 },
          frameRate: { ideal: 120, max: 240 },
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
          this.updateCanvasSize(true);
          window.addEventListener('resize', () => this.updateCanvasSize());
          window.addEventListener('orientationchange', () => this.updateCanvasSize());
          resolve();
          this.startFaceTracking();
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

        this.updateCanvasSize();

        if (this.videoElement.nativeElement.currentTime !== this.lastVideoTime) {
          this.lastVideoTime = this.videoElement.nativeElement.currentTime;
          const video = this.videoElement.nativeElement;
          const results = this.faceLandmarker.detectForVideo(
            video,
            performance.now()
          );

          ctx.clearRect(0, 0, this.canvasWidth, this.canvasHeight);

          if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            this.noLipsDetectedCount = 0;
            this.areLipsVisible = true;

            if (this.isRecording && !this.isFrameCollectionComplete) {
              this.currentFrameCount++;
              this.frameCount = this.currentFrameCount;
              if (this.currentFrameCount >= this.REQUIRED_FRAMES) {
                this.isFrameCollectionComplete = true;
                this.stopRecording();
                return;
              }
            }

            for (const landmarks of results.faceLandmarks) {
              const mirroredLandmarks = landmarks.map((point: { x: number; y: number; z: number }) => ({
                ...point,
                x: 1 - point.x
              }));

              drawingUtils.drawConnectors(
                mirroredLandmarks,
                FaceLandmarker.FACE_LANDMARKS_LIPS,
                { color: "#FFFFFF", lineWidth: 2 }
              );

              const lipPoints = FaceLandmarker.FACE_LANDMARKS_LIPS.flat();
              for (const index of lipPoints) {
                const point = mirroredLandmarks[index];
                if (!point) continue;
                ctx.beginPath();
                ctx.arc(
                  point.x * this.canvasWidth,
                  point.y * this.canvasHeight,
                  2,
                  0,
                  2 * Math.PI
                );
                ctx.fillStyle = '#FF3030';
                ctx.fill();
              }
            }
          } else {
            this.noLipsDetectedCount++;
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
      this.close.emit();
    }, 300);
  }

  startRecording() {
    if (!this.mediaStream) return;

    this.chunks = [];
    this.isRecording = true;
    this.currentFrameCount = 0;
    this.isFrameCollectionComplete = false;
    
    this.mediaRecorder = new MediaRecorder(this.mediaStream, {
      mimeType: 'video/webm;codecs=vp9',
      videoBitsPerSecond: 2500000
    });

    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        this.chunks.push(event.data);
      }
    };

    this.mediaRecorder.onstop = () => {
      this.handleRecordingComplete();
    };

    this.mediaRecorder.start();
  }

  stopRecording() {
    if (!this.isRecording) return;
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
      this.isRecording = false;
      if (this.showCameraModal) {
        this.closeCameraModal();
      }
    }
  }

  private handleRecordingComplete() {
    this.videoBlob = new Blob(this.chunks, { type: 'video/webm' });
    if (this.videoBlob) {
      this.isLoading = true;
      this.errorMessage = null;
      this.videoRecorded.emit(this.videoBlob);
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
      .subscribe({
        next: ({ phrase, confidence }) => {
          this.predictedPhrase = phrase;
          this.predictionConfidence = confidence;
          this.isLoading = false;
          this.errorMessage = null;
        },
        error: (error) => {
          console.error('Prediction failed:', error);
          this.predictedPhrase = 'Prediction failed';
          this.predictionConfidence = null;
          this.isLoading = false;
          this.errorMessage = 'Prediction failed. Please try again with a clearer or longer video.';
        }
      });
  }

  ngOnDestroy() {
    this.stopCamera();
  }

  updateCanvasSize = (forceVideoSize = false) => {
    if (this.videoElement && this.videoElement.nativeElement) {
      const video = this.videoElement.nativeElement;
      const width = (forceVideoSize && video.videoWidth) ? video.videoWidth : (video.videoWidth || this.canvasWidth);
      const height = (forceVideoSize && video.videoHeight) ? video.videoHeight : (video.videoHeight || this.canvasHeight);
      this.canvasWidth = width;
      this.canvasHeight = height;
      if (this.canvas && this.canvas.nativeElement) {
        this.canvas.nativeElement.width = width;
        this.canvas.nativeElement.height = height;
      }
    }
  }
} 