/* tslint:disable:no-unused-variable */

import { TestBed, async, inject } from '@angular/core/testing';
import { CsvReaderService } from './csv-reader.service';

describe('CsvReaderService', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [CsvReaderService]
    });
  });

  it('should ...', inject([CsvReaderService], (service: CsvReaderService) => {
    expect(service).toBeTruthy();
  }));
});
