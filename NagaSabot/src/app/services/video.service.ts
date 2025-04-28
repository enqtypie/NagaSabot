import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, map } from 'rxjs';
import { VideoResult } from '../video-result/video-result.component';

@Injectable({
  providedIn: 'root'
})
export class VideoService {
  private apiUrl = 'http://localhost:5000'; // Flask backend URL

  constructor(private http: HttpClient) {}

  uploadVideo(videoFile: File): Observable<VideoResult> {
    const formData = new FormData();
    formData.append('video', videoFile);

    return this.http.post<any>(`${this.apiUrl}/upload`, formData).pipe(
      map(response => ({
        videoUrl: response.videoUrl,
        phrase: response.phrase || 'No phrase detected',
        accuracy: response.accuracy || 0,
        timestamp: response.timestamp || Date.now()
      }))
    );
  }
} 