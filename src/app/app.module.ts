import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterModule, Routes } from '@angular/router';
import { CollapseModule } from 'ng2-bootstrap';

import { AppComponent } from './app.component';
import { MainComponent } from './main/main.component';
import { ScrapingComponent } from './scraping/scraping.component';

const appRoutes: Routes = [
  { path: 'main', component: MainComponent },
  { path: 'scraping', component: ScrapingComponent },
  { path: '', redirectTo: '/main', pathMatch: 'full' },
  { path: '**', redirectTo: '/main', pathMatch: 'full' }
];

@NgModule({
  declarations: [
    AppComponent,
    MainComponent,
    ScrapingComponent
  ],
  imports: [
    BrowserModule,
    FormsModule,
    RouterModule.forRoot(appRoutes),
    CollapseModule.forRoot()
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
