import { Injectable } from '@angular/core';
import { Http, Response } from '@angular/http';
import { Observable } from 'rxjs';
import { ZipService } from './zip.service';

@Injectable()
export class JsonReaderService {

  constructor(private http: Http, private zip: ZipService) {}

  readJson(jsonFileUrl: string): Observable<any> {
    if (jsonFileUrl.endsWith('.json.zip')) {
      return Observable.fromPromise(
        this.zip.getContentFirstFile(jsonFileUrl, (value: String) => value.endsWith('.json'))
      ).map((data: string) => JSON.parse(data))
        .catch(JsonReaderService.handleError);
    } else if (jsonFileUrl.endsWith('.json')) {
      return this.http.get(jsonFileUrl)
        .map((res: Response) => res.json())
        .catch(JsonReaderService.handleError);
    } else {
      console.error('The url to the json file need to end with ".json" extension or ".json.zip" to be handle by the service.');
      return null;
    }
  }

  private static handleError(error: any): string {
    let errMsg = (error.message) ? error.message :
      error.status ? `${error.status} - ${error.statusText}` : 'Server error';
    console.error(errMsg, error); // log to console instead
    return errMsg;
  }

}
