/* tslint:disable:no-unused-variable */

import { TestBed, async, inject } from '@angular/core/testing';
import { JsonReaderService } from './json-reader.service';

describe('JsonReaderService', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [JsonReaderService]
    });
  });

  it('should ...', inject([JsonReaderService], (service: JsonReaderService) => {
    expect(service).toBeTruthy();
  }));
});
