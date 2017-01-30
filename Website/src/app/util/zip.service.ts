import { Injectable } from '@angular/core';
declare let JSZip:any;
declare let JSZipUtils:any;

@Injectable()
export class ZipService {

  constructor() { }

  load(urlZipFile: string): Promise<ArrayBuffer|string> {
    return new JSZip.external.Promise((resolve, reject) => {
      JSZipUtils.getBinaryContent(urlZipFile,
        (err, data: ArrayBuffer|String) => {
          if (err) {
            reject(err);
          } else {
            resolve(data);
          }
        });
    });
  }

  getContentFirstFile(
    urlZipFile: string,
    fileFilter: (value: string, index: number, array: string[]) => any = () => true,
    type: string = 'string',
    downloadCallback: (data: any) => any = () => {}
    ): Promise<any> {
    return this.load(urlZipFile)
      .then((data) =>
        JSZip.loadAsync(data).then((zip) => {
          downloadCallback(zip);
          return zip.file(Object.keys(zip.files).filter(fileFilter)[0]).async(type);
        })
      );
  }
}
