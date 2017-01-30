/* tslint:disable:no-unused-variable */
import { async, ComponentFixture, TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';
import { DebugElement } from '@angular/core';

import { Lausanne2016Component } from './lausanne-2016.component';

describe('Lausanne2016Component', () => {
  let component: Lausanne2016Component;
  let fixture: ComponentFixture<Lausanne2016Component>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ Lausanne2016Component ]
    })
      .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(Lausanne2016Component);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
