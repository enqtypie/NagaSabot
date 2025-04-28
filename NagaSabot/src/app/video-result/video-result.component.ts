import { Component, Input, Output, EventEmitter, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { VideoService } from '../services/video.service';

export interface VideoResult {
  videoUrl: string;
  phrase: string;
  accuracy: number;
  timestamp: number;
}

@Component({
  selector: 'app-video-result',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './video-result.component.html'
})
export class VideoResultComponent implements OnInit {
  @Input() videoBlob!: Blob;
  @Output() restart = new EventEmitter<void>();
  
  result: VideoResult = {
    videoUrl: '',
    phrase: 'Processing video...',
    accuracy: 0,
    timestamp: Date.now()
  };

  isLoading = true;
  error: string | null = null;

  constructor(private videoService: VideoService) {}

  ngOnInit() {
    console.log('VideoResultComponent initialized with blob:', this.videoBlob);
    this.uploadVideo();
  }

  private uploadVideo() {
    if (!this.videoBlob) {
      console.error('No video blob provided');
      this.error = 'No video data available';
      this.isLoading = false;
      return;
    }

    this.isLoading = true;
    this.error = null;
    this.result.videoUrl = URL.createObjectURL(this.videoBlob);
    console.log('Created video URL:', this.result.videoUrl);
    
    const file = new File([this.videoBlob], 'recorded-video.webm', { type: 'video/webm' });
    console.log('Created file from blob:', file);
    
    this.videoService.uploadVideo(file).subscribe({
      next: (response) => {
        console.log('Upload response:', response);
        this.result = {
          ...this.result,
          phrase: response.phrase,
          accuracy: response.accuracy,
          timestamp: response.timestamp
        };
        this.isLoading = false;
      },
      error: (error) => {
        console.error('Error uploading video:', error);
        this.error = 'Error processing video. Please try again.';
        this.isLoading = false;
      }
    });
  }

  tryAgain() {
    if (this.result.videoUrl) {
      URL.revokeObjectURL(this.result.videoUrl);
    }
    this.restart.emit();
  }

  get accuracyPercentage(): string {
    return `${(this.result.accuracy * 100).toFixed(1)}%`;
  }
} 