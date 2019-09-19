
shinyUI(fluidPage(
  
  titlePanel("Capstone Project - Next world prediction"),
  tabsetPanel(type='tab',
              tabPanel("Application",
                       sidebarLayout(
                         
                         sidebarPanel(
                           textInput('userInput',label="Digit here:",value="Let's go..."),
                           
                           br(),
                           helpText("Digit your phare and you will see prediction")),

                         mainPanel(
                           h4("TOP 5 PREDICTION:"),
                           plotOutput(outputId = "hist") 
                           #tableOutput('guess')
                           )
                       )
              )
  ),
  hr(),
  h4("F Canuto 19/09/2019",
     a(p("Github Repo."), href="")
     )
))