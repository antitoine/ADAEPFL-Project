import { Injectable } from '@angular/core';
import { Http, Response } from '@angular/http';
import { Observable } from 'rxjs';

@Injectable()
export class JsonReaderService {

  constructor(private http: Http) {}

  readJsonData(jsonFileUrl: string): Observable<any> {
    return this.http.get(jsonFileUrl)
      .map(this.extractData)
      .catch(this.handleError);
  }

  private extractData(res: Response): any[] {
    return res.json();
  }

  private handleError(error: any): string {
    let errMsg = (error.message) ? error.message :
      error.status ? `${error.status} - ${error.statusText}` : 'Server error';
    console.error(errMsg); // log to console instead
    return errMsg;
  }

}
