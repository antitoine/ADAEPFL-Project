import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterModule, Routes } from '@angular/router';
import { HttpModule } from '@angular/http';
import { ChartsModule } from 'ng2-charts';
import { CollapseModule, TabsModule } from 'ng2-bootstrap';

import { AppComponent } from './app.component';
import { MainComponent } from './main/main.component';
import { ScrapingComponent } from './scraping/scraping.component';
import { Lausanne2016Component } from './lausanne2016/lausanne2016.component';
import { CsvReaderService } from './csv-reader.service';
import { MapToIterablePipe } from './map-to-iterable.pipe';
import { Lausanne19992016Component } from './lausanne-1999-2016/lausanne-1999-2016.component';

const appRoutes: Routes = [
  { path: 'main', component: MainComponent },
  { path: 'scraping', component: ScrapingComponent },
  { path: 'lausanne2016', component: Lausanne2016Component },
  { path: 'lausanne-1999-2016', component: Lausanne19992016Component },
  { path: '', redirectTo: '/main', pathMatch: 'full' },
  { path: '**', redirectTo: '/main', pathMatch: 'full' }
];

@NgModule({
  declarations: [
    AppComponent,
    MainComponent,
    ScrapingComponent,
    Lausanne2016Component,
    MapToIterablePipe,
    Lausanne19992016Component,
    Lausanne19992016Component
  ],
  imports: [
    BrowserModule,
    FormsModule,
    RouterModule.forRoot(appRoutes),
    ChartsModule,
    CollapseModule.forRoot(),
    TabsModule.forRoot(),
    HttpModule
  ],
  providers: [
    CsvReaderService
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
