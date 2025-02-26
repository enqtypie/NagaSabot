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
  templateUrl: './video-result.component.html',
  styleUrls: ['./video-result.component.css']
})
export class VideoResultComponent implements OnInit {
  @Input() videoBlob!: Blob;
  @Output() restart = new EventEmitter<void>();
  
  result: VideoResult = {
    videoUrl: '',
    phrase: 'Uploading video...',
    accuracy: 0,
    timestamp: Date.now()
  };

  constructor(private videoService: VideoService) {}

  ngOnInit() {
    this.uploadVideo();
  }

  private uploadVideo() {
    // Create local preview URL
    this.result.videoUrl = URL.createObjectURL(this.videoBlob);
    
    // Create file and upload
    const file = new File([this.videoBlob], 'recorded-video.webm', { type: 'video/webm' });
    
    this.videoService.uploadVideo(file).subscribe({
      next: (response) => {
        this.result = {
          ...this.result,
          phrase: response.phrase,
          accuracy: response.accuracy,
          timestamp: response.timestamp
        };
      },
      error: (error) => {
        console.error('Error uploading video:', error);
        this.result.phrase = 'Error uploading video. Please try again.';
      }
    });
  }

  tryAgain() {
    URL.revokeObjectURL(this.result.videoUrl);
    this.restart.emit();
  }
} 