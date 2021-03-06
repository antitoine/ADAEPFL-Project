import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterModule, Routes } from '@angular/router';
import { HttpModule } from '@angular/http';
import { CollapseModule, TabsModule } from 'ng2-bootstrap';

import { CsvReaderService } from './util/csv-reader.service';
import { MapToIterablePipe } from './util/map-to-iterable.pipe';

import { AppComponent } from './app.component';
import { MainComponent } from './main/main.component';
import { ScrapingComponent } from './scraping/scraping.component';
import { Lausanne2016Component } from './lausanne-2016/lausanne-2016.component';
import { Lausanne19992016Component } from './lausanne-1999-2016/lausanne-1999-2016.component';
import { RunnersComponent } from './runners/runners.component';
import { JsonReaderService } from './util/json-reader.service';
import { PlotlyComponent } from './util/plotly/plotly.component';
import { RangePipe } from './util/range.pipe';
import { ZipService } from './util/zip.service';
import { SlimLoadingBarModule } from 'ng2-slim-loading-bar';
import { AboutComponent } from './about/about.component';

const appRoutes: Routes = [
  { path: 'main', component: MainComponent },
  { path: 'scraping', component: ScrapingComponent },
  { path: 'lausanne-2016', component: Lausanne2016Component },
  { path: 'lausanne-1999-2016', component: Lausanne19992016Component },
  { path: 'runners', component: RunnersComponent },
  { path: 'about', component: AboutComponent },
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
    Lausanne19992016Component,
    RunnersComponent,
    PlotlyComponent,
    RangePipe,
    AboutComponent
  ],
  imports: [
    BrowserModule,
    FormsModule,
    RouterModule.forRoot(appRoutes),
    CollapseModule.forRoot(),
    TabsModule.forRoot(),
    HttpModule,
    SlimLoadingBarModule.forRoot()
  ],
  providers: [
    CsvReaderService,
    JsonReaderService,
    ZipService
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
