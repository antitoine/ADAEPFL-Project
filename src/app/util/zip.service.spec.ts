/* tslint:disable:no-unused-variable */

import { TestBed, async, inject } from '@angular/core/testing';
import { ZipService } from './zip.service';

describe('ZipService', () => {
  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [ZipService]
    });
  });

  it('should ...', inject([ZipService], (service: ZipService) => {
    expect(service).toBeTruthy();
  }));
});
