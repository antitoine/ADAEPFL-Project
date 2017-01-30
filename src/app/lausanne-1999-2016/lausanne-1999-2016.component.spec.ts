/* tslint:disable:no-unused-variable */
import { async, ComponentFixture, TestBed } from '@angular/core/testing';
import { By } from '@angular/platform-browser';
import { DebugElement } from '@angular/core';

import { Lausanne19992016Component } from './lausanne-1999-2016.component';

describe('Lausanne19992016Component', () => {
  let component: Lausanne19992016Component;
  let fixture: ComponentFixture<Lausanne19992016Component>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ Lausanne19992016Component ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(Lausanne19992016Component);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
