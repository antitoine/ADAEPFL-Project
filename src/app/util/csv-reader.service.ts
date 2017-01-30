import { Injectable } from '@angular/core';
import { Response, Http } from "@angular/http";
import { Observable } from 'rxjs';
import 'rxjs/Rx';

@Injectable()
export class CsvReaderService {

  constructor(private http: Http) {}

  readCsvData(csvFileUrl: string): Observable<any> {
    return this.http.get(csvFileUrl)
      .map(this.extractData)
      .catch(this.handleError);
  }

  getColumn(csvData: any[], column: any): any[] {
    let n = csvData[0].findIndex(element => element === column);
    if (n == -1) {
      return null;
    }
    return csvData.map(x => x[n]).slice(1);
  }

  private extractData(res: Response): any[] {
    let csvData = res['_body'] || '';
    let allTextLines = csvData.split(/\r\n|\n/);
    let headers = allTextLines[0].split(',');
    let lines = [];

    for ( let i = 0; i < allTextLines.length; i++) {
      // split content based on comma
      let data = allTextLines[i].split(',');
      if (data.length == headers.length) {
        let tarr = [];
        for ( let j = 0; j < headers.length; j++) {
          tarr.push(data[j]);
        }
        lines.push(tarr);
      }
    }
    return lines;
  }

  private handleError(error: any): string {
    let errMsg = (error.message) ? error.message :
      error.status ? `${error.status} - ${error.statusText}` : 'Server error';
    console.error(errMsg); // log to console instead
    return errMsg;
  }

}
