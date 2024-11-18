using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class PauseMenu : MonoBehaviour
{
    public string mainMenuName;
    public GameObject pauseScreen;
    private bool isPaused;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if(Input.GetKeyDown(KeyCode.Escape)) {
            PauseUnPause();
        }
    }

    public void MainMenu() {
        SceneManager.LoadScene(mainMenuName);
    }

    public void PauseUnPause(){
        if(isPaused) {
            isPaused = false;
            pauseScreen.SetActive(false);
        }
        else {
            isPaused = true;
            pauseScreen.SetActive(true);
        }
    }
}
